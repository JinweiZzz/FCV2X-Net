import argparse
import json
import os
import math
import pickle
import random
import networkx as nx
import sys
import time
from collections import defaultdict, deque
from copy import deepcopy
import signal
import setproctitle
from pyproj import Proj

import numpy as np
import torch
import torch.nn.functional as F
from moss import Engine, TlPolicy, Verbosity
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dqn_utils.models import R_Actor, VR_Actor
from dqn_utils.utils.config import get_config
from moss.export import DBRecorder
from torch.optim.lr_scheduler import LambdaLR

torch.set_float32_matmul_precision('medium')

def decompose_action(x, sizes):
    out = []
    for i in sizes:
        x, r = divmod(x, i)
        out.append(r)
    return out

def _t2n(x):
    return x.detach().cpu().numpy()

class Env:
    def __init__(self, data_path, step_size, step_count, log_dir, reward, base_algo, tl_interval, yellow_time=0, save=False, record=0, args=None, reward_weight=1):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir+'details'):  
            os.makedirs(self.log_dir+'details', exist_ok=True)
        MAP_PATH = f'{data_path}/map.pb'
        TRIP_PATH = f'{data_path}/persons.pb'
        self.eng = Engine(
            name='Routing',
            map_file=MAP_PATH,
            person_file=TRIP_PATH,
            start_step=0,
            step_interval=1,
            seed=43,
            verbose_level=Verbosity.NO_OUTPUT,
            person_limit=-1,
            junction_yellow_time=1,
            phase_pressure_coeff=1.5,
            speed_stat_interval=0,
            out_xmin=-1e999,
            out_ymin=-1e999,
            out_xmax=1e999,
            out_ymax=1e999,
            device=0
        )
        self.record = record
        if record == 1:
            self.recorder = DBRecorder(
                self.eng,
                "postgres://sim:tnBB0Yf4tm2fIrUi1KB2LqqZTqyrztSa@pg-fib.db.fiblab.tech/simulation",
                "srt.map_grid_network1",  # map collection used for webui-backend
                "south_beijing",
            )  # used for PostgreSQL output (optional)
        
        print('Engines created!')
        M = self.eng.get_map(False)
        self.M = M
        self.proj=Proj(M.header.projection)
        vehs = self.eng.get_persons(False)

        self.num_agents = len(vehs.persons)
        self.veh_dest_lanes = [m.schedules[0].trips[0].end.lane_position.lane_id for m in vehs.persons]
        self.veh_dest_roads = np.array([m.schedules[0].trips[0].routes[0].driving.road_ids[-1] for m in vehs.persons])  
        max_veh_id = max([m.id for m in vehs.persons])
        self.veh_id2idxs = np.zeros(max_veh_id+1, dtype=int)

        for i, m in enumerate(vehs.persons):
            self.veh_id2idxs[m.id] = i

        self.target_vehicle_id = args.target_vehicle_id
        self.target_vehicle_idx = self.veh_id2idxs[self.target_vehicle_id]

        self.lane_id2idxs = {}
        for i, lane in enumerate(M.lanes):
            self.lane_id2idxs[lane.id] = i
        self.lane_id2idxs = self.lane_id2idxs
        lane_start_xs = np.array([M.lanes[i].center_line.nodes[0].x for i in range(len(M.lanes))])
        lane_start_ys = np.array([M.lanes[i].center_line.nodes[0].y for i in range(len(M.lanes))])
        lane_end_xs = np.array([M.lanes[i].center_line.nodes[-1].x for i in range(len(M.lanes))])
        lane_end_ys = np.array([M.lanes[i].center_line.nodes[-1].y for i in range(len(M.lanes))])
        self.lane_start_xs, self.lane_start_ys = lane_start_xs, lane_start_ys
        self.road_start_xs = np.array([lane_start_xs[self.lane_id2idxs[road.lane_ids[0]]] for road in M.roads])
        self.road_start_ys = np.array([lane_start_ys[self.lane_id2idxs[road.lane_ids[0]]] for road in M.roads])
        self.road_end_xs = np.array([lane_end_xs[self.lane_id2idxs[road.lane_ids[-1]]] for road in M.roads])
        self.road_end_ys = np.array([lane_end_ys[self.lane_id2idxs[road.lane_ids[-1]]] for road in M.roads])
        self.veh2dest = {veh.id: veh.schedules[0].trips[0].routes[0].driving.road_ids[-1] for veh in vehs.persons}

        self.source_state_dim = 2
        self.neighbor_state_dim = 2
        self.edge_dim = 2

        if os.path.exists(f'{data_path}/selected_person_ids.json'):
            with open(f'{data_path}/selected_person_ids.json', 'r') as f:       ### 只对这些vehs做observe和routing
                self.selected_person_ids = json.load(f)
        else:
            self.selected_person_ids = list(self.veh2dest.keys())

        self.selected_person_masks = np.array([0]*len(self.veh2dest))
        for i, veh in enumerate(self.veh2dest):
            if veh in self.selected_person_ids:
                self.selected_person_masks[i] = 1

        self.vehicles = {}
        self._step = 0
        self.eng.set_tl_duration_batch(list(range(self.eng.junction_count)), tl_interval)
        self.eng.set_tl_policy_batch(list(range(self.eng.junction_count)), base_algo)

        self._cid = self.eng.make_checkpoint()
        self.step_size = step_size
        self.step_count = step_count
        self.reward = reward
        self.info = {
            'ATT': 1e999,
            'Throughput': 0,
            'rewards': 0,
            'distance_reward': 0,
            'time_reward': 0,
            'ATT_finished': 1e999,
            'VEH': 0
        }
        self.success_travel_time, self.success_travel = 0, 0
        self.total_travel_time, self.total_travel = 0, 0
        self.data_path = data_path

        ## 对所有的路学习相接
        laneid2features = {}
        for lane in M.lanes:
            laneid2features[lane.id] = lane

        road2adjroad = {}
        for lane in M.lanes:
            if str(lane.parent_id)[0] == '3':     #junction
                continue
            for lane_successor in lane.successors:
                if str(laneid2features[lane_successor.id].parent_id)[0] == '3':     #junction
                    tmp = laneid2features[laneid2features[lane_successor.id].successors[0].id].parent_id
                    if str(tmp)[0] == '3' or tmp == lane.parent_id:
                        continue
                    if lane.parent_id not in road2adjroad:
                        road2adjroad[lane.parent_id] = set([tmp])
                    else:
                        road2adjroad[lane.parent_id].add(tmp)
        for road in road2adjroad:
            road2adjroad[road] = list(road2adjroad[road])

        self.max_action_size = max([len(road2adjroad[road]) for road in road2adjroad])

        road2preroad = {}
        for road in road2adjroad:
            for adjroad in road2adjroad[road]:
                if adjroad not in road2preroad:
                    road2preroad[adjroad] = set([road])
                else:
                    road2preroad[adjroad].add(road)
        for road in road2preroad:
            road2preroad[road] = list(road2preroad[road])

        self.road2adjroad = road2adjroad
        available_actions = {}
        for road in road2adjroad:
            available_actions[road] = np.zeros(self.max_action_size, dtype=int)
            available_actions[road][:len(road2adjroad[road])] = 1
            available_actions[road] = available_actions[road].tolist()
        self.available_actions = available_actions

        ### 对road之间构图，需要同步更新所有road的状态。这样的话需要对road进行重编号

        lane_list = []
        road_length = []
        roadid2distance = {}
        self.road_id2idxs, self.road_idxs2id = {}, {}
        for idx, road in enumerate(M.roads):
            lane_list.append(road.lane_ids)
            road_length.append(M.lanes[self.lane_id2idxs[road.lane_ids[0]]].length)
            roadid2distance[road.id] = M.lanes[self.lane_id2idxs[road.lane_ids[0]]].length
            self.road_id2idxs[road.id] = idx
            self.road_idxs2id[idx] = road.id
        
        self.roadidx2adjroadidx = {}
        for road in road2adjroad:
            self.roadidx2adjroadidx[self.road_id2idxs[road]] = [self.road_id2idxs[adjroad] for adjroad in road2adjroad[road]]

        def lane_numpy(lanes_list):
            max_lane_num = max([len(i) for i in lanes_list])
            lanes_array = np.zeros((len(lanes_list), max_lane_num), dtype=int)
            lanes_zero = np.zeros((len(lanes_list), max_lane_num), dtype=int)
            for i, lanes in enumerate(lanes_list):
                lanes_array[i, :len(lanes)] = lanes
                lanes_zero[i, len(lanes):] = 1
            return lanes_array, lanes_zero

        self.lanes_list, self.lanes_zero = lane_numpy(lane_list) 

        self.road_length = np.array(road_length)

        def calculate_angle(x1, y1, x2, y2, x, y):
            dx1 = np.array(x2 - x1)
            dy1 = np.array(y2 - y1)
            dx2 = np.array(x - x2)
            dy2 = np.array(y - y2)
            theta_A_rad = math.degrees(math.atan2(dy1, dx1))
            theta_B_rad = np.array([math.degrees(math.atan2(dy2[i], dx2[i])) for i in range(len(dx2))])
            angle = abs((theta_B_rad - theta_A_rad) % 360)
            return angle
        
        ### 以junction为点，road为边构图
        self.junctionid2idxs = {}
        self.junctionidx2id = {}
        for i, junction in enumerate(M.junctions):
            self.junctionid2idxs[junction.id] = i
            self.junctionidx2id[i] = junction.id

        junction2inroad = {}
        junction2outroad = {}
        for junction in M.junctions:
            in_road_ids = []
            out_road_ids = []
            for driving_lane_groups in junction.driving_lane_groups:
                in_road_ids.append(driving_lane_groups.in_road_id)
                out_road_ids.append(driving_lane_groups.out_road_id)
            junction2inroad[self.junctionid2idxs[junction.id]] = list(set(in_road_ids))
            junction2outroad[self.junctionid2idxs[junction.id]] = list(set(out_road_ids))
        
        road2injunction = {}
        road2outjunction = {}
        for junction in junction2inroad:
            for road in junction2inroad[junction]:
                if road not in road2injunction:
                    road2injunction[road] = [junction]
                else:
                    road2injunction[road].append(junction)
            for road in junction2outroad[junction]:
                if road not in road2outjunction:
                    road2outjunction[road] = [junction]
                else:
                    road2outjunction[road].append(junction)
        for road in road2injunction:
            road2injunction[road] = list(set(road2injunction[road]))
        for road in road2outjunction:
            road2outjunction[road] = list(set(road2outjunction[road]))

        ### 对路口构造边
        edge_set = []
        for road in M.roads:
            road_id = road.id
            source_junction = road2outjunction[road_id][0]
            target_junction = road2injunction[road_id][0]
            edge_set.append((source_junction, target_junction, self.road_length[self.road_id2idxs[road_id]]))

        G = nx.DiGraph()
        G.add_weighted_edges_from(edge_set)
        
        ### 提前存储所有路和所有终点之间的关系
        self.angle_matrix = np.zeros((len(M.roads), len(M.roads)))
        self.distance_matrix = np.ones((len(M.roads), len(M.roads)))*(-1000)
        shortest_distance = {}
        for i, road in enumerate(M.roads):
            ### 
            self.angle_matrix[i, :] = calculate_angle(self.road_start_xs[i], self.road_start_ys[i], self.road_end_xs[i], self.road_end_ys[i], self.road_start_xs, self.road_start_ys)
            for j, end_road in enumerate(M.roads):
                if i == j:
                    self.distance_matrix[i, j] = 0
                    continue
                self.distance_matrix[i, j] = nx.shortest_path_length(G, road2injunction[self.road_idxs2id[i]][0], road2outjunction[self.road_idxs2id[j]][0], weight='weight')
                shortest_path = nx.shortest_path(G, road2injunction[self.road_idxs2id[i]][0], road2outjunction[self.road_idxs2id[j]][0], weight='weight')
                shortest_path = [self.junctionidx2id[junction] for junction in shortest_path]
                shortest_distance[(road.id, end_road.id)] = shortest_path
                self.distance_matrix[i, j] += road_length[i]

        ## 存储道路在不同时间步的属性
        self.road_states = np.zeros((self.step_count+1, len(M.roads), self.neighbor_state_dim))

        ### 保存一个
        self.road_adj_mask = np.zeros((len(M.roads), len(M.roads)))
        for road in road2adjroad:
            self.road_adj_mask[self.road_id2idxs[road], self.road_id2idxs[road]] = 1
            for adjroad in road2adjroad[road]:
                self.road_adj_mask[self.road_id2idxs[road], self.road_id2idxs[adjroad]] = 1
                for two_hop_adjroad in road2adjroad[adjroad]:
                    self.road_adj_mask[self.road_id2idxs[road], self.road_id2idxs[two_hop_adjroad]] = 1

        ### 初始states
        self.fresh_state()
        self.roadidx2corrroadidx = self.roadidx2adjroadidx

        self.reward_weight = reward_weight
        self.balancing_coef = args.balancing_coef

        self.routing_queries = []

        self.agg = args.agg
        self.corr_agg = args.corr_agg
        self.lc_interval = args.lc_interval

    def lc_edge_extract(self):
        road_count = torch.tensor(self.road_states[:self._step, :, 0]).T        ### num_roads*step_count
        road_ffn = torch.fft.rfft(road_count, norm='ortho', dim=-1)     ### num_roads*step_count//2+1

        road_ffn = torch.cat([road_ffn.real, road_ffn.imag], dim=-1)        ### num_roads*(step_count+2)

        sim = road_ffn @ road_ffn.T
        sim = sim / (torch.norm(road_ffn, dim=-1).unsqueeze(-1) @ torch.norm(road_ffn, dim=-1).unsqueeze(0))        ### num_roads * num_roads

        ### 将2-hop内邻居mask掉
        sim[self.road_adj_mask==1] = -1e999

        ### 找到topk邻居
        topk = 4
        _, topk_idx = torch.topk(sim, topk, dim=-1)
        topk_idx = topk_idx.cpu().numpy()

        ### 存储dynamic edge
        self.road2corrroad = {}  
        self.roadidx2corrroadidx = {}
        for i, road in enumerate(self.road_idxs2id):
            self.road2corrroad[road] = [self.road_idxs2id[j] for j in topk_idx[i]]
            self.roadidx2corrroadidx[i] = topk_idx[i].tolist()
        
        # with open('{}/road2corrroad_{}.json'.format(self.data_path, self._step), 'w') as f:
        #     json.dump(self.road2corrroad, f)
        
        # with open('{}/similarity_{}.json'.format(self.data_path, self._step), 'w') as f:
        #     json.dump(sim.cpu().numpy().tolist(), f)

    def add_env_vc(self, vid, road, time, destination):
        self.vehicles[vid]={
            "destination": destination,
            "start_time":time,
            "time": time,
            "road":road,
            "next_road":None,
            "reward":None,
            "state":None,
            "is_new":True,
            'done':False,
            'last_reward':False,
            'last_road': None
            }

    def reset(self):
        self.eng.restore_checkpoint(self._cid)
        self.vehicles = {}
        self.success_travel_time, self.success_travel = 0, 0
        self.total_travel_time, self.total_travel = 0, 0

    def get_vehicle_distances(self):
        info = self.eng.fetch_persons()
        xs, ys = info['x'], info['y']
        lane_xs, lane_ys = self.lane_start_xs[info['lane_id'].tolist()], self.lane_start_ys[info['lane_id'].tolist()]

        distances = np.sqrt((xs - lane_xs)**2 + (ys - lane_ys)**2)
        return distances
    
    def get_routing_demand_ids(self):
        m = self.get_vehicle_distances()   
        n = self.eng.fetch_persons()
        m[(n['status']!=2)|(n['lane_parent_id']>=300000000)] = 1e999
        m[self.selected_person_masks==0] = 1e999
        roads = n['lane_parent_id'][m<60]
        ids = n['id'][m<60]
        idxs = self.veh_id2idxs[ids]

        routing_types = [[] for _ in range(2)]  # 0: 需要转向 1: 到达终点
        for veh_id, veh_idx, road in zip(ids, idxs, roads):
            if self.vehicles[veh_id]['last_reward'] is True:
                continue
            if (self.vehicles[veh_id]['next_road'] is not None) and road != self.vehicles[veh_id]['next_road']:  
                continue
            if self.vehicles[veh_id]['destination'] == road:   # 第一次到达终点道路
                routing_types[1].append([veh_id, veh_idx, road])
            else:
                routing_types[0].append([veh_id, veh_idx, road])
        return routing_types
    
    def reach_dest_detect(self, road, dest):
        if dest in self.road2adjroad[road]:
            return True
        return False

    def insert_next_road(self, veh, dest, action=None):
        road = self.eng.fetch_persons()['lane_parent_id'][self.veh_id2idxs[veh]]

        action = self.road2adjroad[road][action]
        if action != dest:
            self.eng.set_vehicle_route(veh, [road, action, dest])
        else:
            self.eng.set_vehicle_route(veh, [road, dest])
        self.vehicles[veh]['next_road'] = action
        self.vehicles[veh]['last_road'] = road
    
    def update_env_vc_info(self,vc,step,road,state, done=False, first_decision=False, last_reward=False):	 	 
        ### 更新此时决策的信息
        self.vehicles[vc]["time"]=step
        self.vehicles[vc]["road"]=road
        self.vehicles[vc]["state"]=state
        self.vehicles[vc]["is_new"]=False
        self.vehicles[vc]['done'] = done
        self.vehicles[vc]['first_decision']=first_decision
        self.vehicles[vc]['last_reward']=last_reward
    
    def get_new_vehs(self, step):
        ### 查找在step和step-1之间出发的agent
        person_info = self.eng.fetch_persons()
        departure_times = person_info['departure_time']
        ids = person_info['id'][(departure_times>=(step-1))&(departure_times<step)]
        roads = person_info['lane_parent_id'][(departure_times>=(step-1))&(departure_times<step)]
        departure_times = person_info['departure_time'][(departure_times>=(step-1))&(departure_times<step)]
        selected_ids = [id in self.selected_person_ids for id in ids]
        ids = ids[selected_ids]
        roads = roads[selected_ids]
        departure_times = departure_times[selected_ids]
        return ids, roads, departure_times
    
    def success_routing(self, vc, timeout=False):
        self.success_travel_time += self._step-self.vehicles[vc]['start_time'] if not timeout else 0
        self.total_travel_time += self._step-self.vehicles[vc]['start_time']
        self.success_travel += 1 if not timeout else 0
        self.total_travel += 1

    def lane_vehicle_cal(self):
        speed_threshold = 0.1
        vehicle_counts = np.zeros(len(self.lane_id2idxs))
        vehicle_waiting_counts = np.zeros(len(self.lane_id2idxs))
        for lane, status, v, lane_parent_id in zip(self.eng.fetch_persons()['lane_id'], self.eng.fetch_persons()['status'], self.eng.fetch_persons()['v'], self.eng.fetch_persons()['lane_parent_id']):
            if status == 2 and lane_parent_id < 300000000:
                vehicle_counts[self.lane_id2idxs[lane]] += 1
                if v < speed_threshold:
                    vehicle_waiting_counts[self.lane_id2idxs[lane]] += 1
        return vehicle_counts, vehicle_waiting_counts
    
    def road_speed_cal(self, road_vehicle_count):
        road_speed_overall = np.zeros(len(self.road_id2idxs))
        for status, v, road in zip(self.eng.fetch_persons()['status'], self.eng.fetch_persons()['v'], self.eng.fetch_persons()['lane_parent_id']):
            if status == 2 and road < 300000000:
                road_speed_overall[self.road_id2idxs[road]] += v
        road_speed_ave = road_speed_overall/road_vehicle_count
        road_travel_time_ave = self.road_length/road_speed_ave

        # 将nan值设为-1
        road_speed_ave[np.isnan(road_speed_ave)] = -1
        road_travel_time_ave[np.isnan(road_travel_time_ave)] = -1

        ### 将travel_time_ave为np.inf的值设为-1
        road_travel_time_ave[np.isinf(road_travel_time_ave)] = -1
        return road_speed_ave, road_travel_time_ave
    
    def fresh_state(self):
        lane_vehicle_counts, lane_vehicle_waiting_counts = self.lane_vehicle_cal()
        road_vehicle = lane_vehicle_counts[self.lanes_list]
        road_vehicle[self.lanes_zero==1]==0
        road_vehicle = np.sum(road_vehicle, axis=1)

        road_waiting_vehicle = lane_vehicle_waiting_counts[self.lanes_list]
        road_waiting_vehicle[self.lanes_zero==1]==0
        road_waiting_vehicle = np.sum(road_waiting_vehicle, axis=1)

        road_vehicle_density = road_vehicle/self.road_length
        road_waiting_vehicle_density = road_waiting_vehicle/self.road_length

        self.road_state = np.vstack([road_vehicle_density, road_waiting_vehicle_density]).T
        self.road_states[self._step] = self.road_state

    ### 道路间夹角\路的距离作为边的特征
    def get_state(self, road, destination):
        state_dim = self.neighbor_state_dim
        state = np.zeros((self.max_action_size*self.source_state_dim+1))
            
        adj_roadidxs = self.roadidx2adjroadidx[self.road_id2idxs[road]]
        road_angle = self.angle_matrix[adj_roadidxs, self.road_id2idxs[destination]]
        road_distance = self.distance_matrix[adj_roadidxs, self.road_id2idxs[destination]]
        road_length = self.road_length[adj_roadidxs]
        state[:len(road_angle)] = road_distance/1000  
        state[self.max_action_size*1:self.max_action_size*1+len(road_distance)] = road_length/100   
        state[-1] = self._step/self.step_count
        
        adj_state = np.zeros((self.max_action_size, state_dim))
        adj_state[:len(adj_roadidxs)] = self.road_state[adj_roadidxs]

        adj_road_neighbor_state, adj_road_neighbor_mask, adj_road_neighbor_dest_angle, adj_road_neighbor_past_angle = None, None, None, None
        if self.agg:
            adj_road_neighbor_state = np.zeros((self.max_action_size, self.max_action_size, state_dim))
            adj_road_neighbor_mask = np.zeros((self.max_action_size, self.max_action_size))
            adj_road_neighbor_dest_angle = np.zeros((self.max_action_size, self.max_action_size))
            adj_road_neighbor_past_angle = np.zeros((self.max_action_size, self.max_action_size))
            for idx, adj_road in enumerate(adj_roadidxs):
                two_hop_adj_roadidxs = self.roadidx2adjroadidx[adj_road]
                adj_road_neighbor_state[idx, :len(two_hop_adj_roadidxs), :] = self.road_state[two_hop_adj_roadidxs]
                adj_road_neighbor_mask[idx, :len(two_hop_adj_roadidxs)] = 1
                adj_road_neighbor_dest_angle[idx, :len(two_hop_adj_roadidxs)] = self.angle_matrix[two_hop_adj_roadidxs, self.road_id2idxs[destination]] - self.angle_matrix[adj_road, self.road_id2idxs[destination]]       # 角度越靠近0越好
                adj_road_neighbor_past_angle[idx, :len(two_hop_adj_roadidxs)] = self.angle_matrix[adj_road, self.road_id2idxs[road]] - self.angle_matrix[two_hop_adj_roadidxs, adj_road]      # 角度越靠近0越好
        
        corr_state, corr_road_neighbor_state, corr_road_neighbor_mask, corr_road_neighbor_dest_angle, corr_road_neighbor_past_angle = None, None, None, None, None
        if self.corr_agg:
            corr_roadidxs = self.roadidx2corrroadidx[self.road_id2idxs[road]]
            corr_state = np.zeros((self.max_action_size, state_dim))
            corr_road_neighbor_state = np.zeros((self.max_action_size, self.max_action_size, state_dim))
            corr_road_neighbor_mask = np.zeros((self.max_action_size, self.max_action_size))
            corr_road_neighbor_dest_angle = np.zeros((self.max_action_size, self.max_action_size))
            corr_road_neighbor_past_angle = np.zeros((self.max_action_size, self.max_action_size))
            corr_state[:len(corr_roadidxs)] = self.road_state[corr_roadidxs]
            for idx, corr_road in enumerate(corr_roadidxs):
                two_hop_corr_roadidxs = self.roadidx2adjroadidx[corr_road]
                corr_road_neighbor_state[idx, :len(two_hop_corr_roadidxs), :] = self.road_state[two_hop_corr_roadidxs]
                corr_road_neighbor_mask[idx, :len(two_hop_corr_roadidxs)] = 1
                corr_road_neighbor_dest_angle[idx, :len(two_hop_corr_roadidxs)] = self.angle_matrix[two_hop_corr_roadidxs, self.road_id2idxs[destination]] - self.angle_matrix[corr_road, self.road_id2idxs[destination]]       # 角度越靠近0越好
                corr_road_neighbor_past_angle[idx, :len(two_hop_corr_roadidxs)] = self.angle_matrix[corr_road, self.road_id2idxs[road]] - self.angle_matrix[two_hop_corr_roadidxs, corr_road]      # 角度越靠近0越好 

        if self.agg == 0 and self.corr_agg == 0:
            state = np.concatenate([state, adj_state.flatten()])
            adj_state = None
        return state, adj_state, adj_road_neighbor_state, adj_road_neighbor_mask, adj_road_neighbor_dest_angle, adj_road_neighbor_past_angle, corr_state, corr_road_neighbor_state, corr_road_neighbor_mask, corr_road_neighbor_dest_angle, corr_road_neighbor_past_angle

    def extra_reward(self, vc, last_road, next_road):
        last_distance = self.distance_matrix[self.road_id2idxs[last_road], self.road_id2idxs[self.vehicles[vc]['destination']]]
        next_distance = self.distance_matrix[self.road_id2idxs[next_road], self.road_id2idxs[self.vehicles[vc]['destination']]]
        reward = self.reward_weight*(last_distance - next_distance)/1000
        return reward

    def step(self, actions=None):
        new_experiences = {'action_side':{}, 'obs_side':{}}
        if len(actions) > 0 and self._step < self.step_count:
            for (veh, dest), action in zip(self.routing_queries, actions):  
                self.insert_next_road(veh, dest, action)      
                new_experience = {veh: {'action_signal': 1, 'action': action}}
                new_experiences['action_side'].update(new_experience)

        self.eng.next_step(1)
        self._step = self._step + 1
        if self.record:
            self.recorder.record()
        
        if self._step % self.lc_interval == 0:       ### 每5分钟更新一次动态边
            self.lc_edge_extract()

        vehs, roads, departure_times = self.get_new_vehs(self._step)
        for veh, road, departure_time in zip(vehs, roads, departure_times):
            destination = self.veh2dest[veh]  
            self.add_env_vc(veh, road, departure_time, destination)

        if self._step % 300 == 0:       ### 每5分钟更新一次动态边
            self.lc_edge_extract()

        self.routing_queries = []

        [routing_demands, finish_demands] = self.get_routing_demand_ids()     
        if len(routing_demands) > 0 or len(finish_demands) > 0:
            self.fresh_state()    

        next_veh, success_veh = [], []
        all_rewards, time_rewards, distance_rewards = 0, 0, 0
        next_states, next_acs, next_adj_states, next_adj_road_neighbor_states, next_adj_road_neighbor_masks, next_adj_road_neighbor_dest_angles, next_adj_road_neighbor_past_angles, next_corr_states, next_corr_road_neighbor_states, next_corr_road_neighbor_masks, next_corr_road_neighbor_dest_angles, next_corr_road_neighbor_past_angles = \
         [], [], [], [], [], [], [], [], [], [], [], []
        
        self.info['reward'] = 0
        self.info['time_reward'] = 0
        self.info['distance_reward'] = 0

        ## 日常决策
        for (vc, vidx, road) in routing_demands:
            if self._step < self.step_count: 
                next_state, next_adj_state, next_adj_road_neighbor_state, next_adj_road_neighbor_mask, next_adj_road_neighbor_dest_angle, next_adj_road_neighbor_past_angle, next_corr_state, next_corr_road_neighbor_state, next_corr_road_neighbor_mask, next_corr_road_neighbor_dest_angle, next_corr_road_neighbor_past_angle = self.get_state(road, self.vehicles[vc]["destination"])
                available_action = self.available_actions[road]
                next_veh.append(vc)
                if self.vehicles[vc]['is_new']:
                    reward = None
                    self.update_env_vc_info(vc, self._step, road, next_state[-1], first_decision=True)
                else:
                    if self.vehicles[vc]['first_decision']:
                        reward = None
                    else:
                        distance_reward = self.extra_reward(vc, self.vehicles[vc]['last_road'], self.vehicles[vc]['next_road'])
                        time_reward = -(self._step-self.vehicles[vc]['time']) / 100
                        if self.reward == 'only_distance':
                            reward = distance_reward
                        elif self.reward == 'distance':
                            reward = time_reward + distance_reward*self.balancing_coef
                        else:
                            reward = time_reward
                        all_rewards += time_reward + distance_reward*self.balancing_coef
                        time_rewards += time_reward
                        distance_rewards += distance_reward
                    self.update_env_vc_info(vc, self._step, road, next_state[-1])
                self.routing_queries.append((vc, self.vehicles[vc]['destination']))
                new_experience = {vc: {'next_state': next_state, 'available_action': available_action, 'reward': reward, 'success': 0, 'timeout': 0, 'action_signal': 0, \
                                       'next_adj_state': next_adj_state, 'next_adj_road_neighbor_state': next_adj_road_neighbor_state, 'next_adj_road_neighbor_mask': next_adj_road_neighbor_mask, \
                                        'next_adj_road_neighbor_dest_angle': next_adj_road_neighbor_dest_angle, 'next_adj_road_neighbor_past_angle': next_adj_road_neighbor_past_angle, \
                                            'next_corr_state': next_corr_state, 'next_corr_road_neighbor_state': next_corr_road_neighbor_state, 'next_corr_road_neighbor_mask': next_corr_road_neighbor_mask, \
                                                'next_corr_road_neighbor_dest_angle': next_corr_road_neighbor_dest_angle, 'next_corr_road_neighbor_past_angle': next_corr_road_neighbor_past_angle}}
                new_experiences['obs_side'].update(new_experience)
                next_states.append(next_state)
                next_adj_states.append(next_adj_state)
                next_adj_road_neighbor_states.append(next_adj_road_neighbor_state)
                next_adj_road_neighbor_masks.append(next_adj_road_neighbor_mask)
                next_adj_road_neighbor_dest_angles.append(next_adj_road_neighbor_dest_angle)
                next_adj_road_neighbor_past_angles.append(next_adj_road_neighbor_past_angle)
                next_corr_states.append(next_corr_state)
                next_corr_road_neighbor_states.append(next_corr_road_neighbor_state)
                next_corr_road_neighbor_masks.append(next_corr_road_neighbor_mask)
                next_corr_road_neighbor_dest_angles.append(next_corr_road_neighbor_dest_angle)
                next_corr_road_neighbor_past_angles.append(next_corr_road_neighbor_past_angle)
                next_acs.append(available_action)

        assert len(next_veh) == len(self.routing_queries) == len(next_states) == len(next_acs)

        for (vc, vidx, road) in finish_demands:  
            next_state, next_adj_state, next_adj_road_neighbor_state, next_adj_road_neighbor_mask, next_adj_road_neighbor_dest_angle, next_adj_road_neighbor_past_angle, next_corr_state, next_corr_road_neighbor_state, next_corr_road_neighbor_mask, next_corr_road_neighbor_dest_angle, next_corr_road_neighbor_past_angle = self.get_state(road, self.vehicles[vc]["destination"])
            success_veh.append(vc)
            if self.vehicles[vc]['first_decision']:
                reward = None
                new_experience = {vc: {'next_state': next_state, 'available_action': None, 'reward': [10], 'success': 1, 'timeout': 0, 'action_signal': 0}}
            else:
                distance_reward = self.extra_reward(vc, self.vehicles[vc]['last_road'], self.vehicles[vc]['next_road'])
                time_reward = -(self._step-self.vehicles[vc]['time']) / 100
                if self.reward == 'only_distance':
                    reward = distance_reward
                elif self.reward == 'distance':
                    reward = time_reward + distance_reward*self.balancing_coef
                else:
                    reward = time_reward
                new_experience = {vc: {'next_state': next_state, 'available_action': None, 'reward': [reward, 10], 'success': 1, 'timeout': 0, 'action_signal': 0,\
                                       'next_adj_state': next_adj_state, 'next_adj_road_neighbor_state': next_adj_road_neighbor_state, 'next_adj_road_neighbor_mask': next_adj_road_neighbor_mask, \
                                        'next_adj_road_neighbor_dest_angle': next_adj_road_neighbor_dest_angle, 'next_adj_road_neighbor_past_angle': next_adj_road_neighbor_past_angle, \
                                            'next_corr_state': next_corr_state, 'next_corr_road_neighbor_state': next_corr_road_neighbor_state, 'next_corr_road_neighbor_mask': next_corr_road_neighbor_mask, \
                                                'next_corr_road_neighbor_dest_angle': next_corr_road_neighbor_dest_angle, 'next_corr_road_neighbor_past_angle': next_corr_road_neighbor_past_angle}}
                all_rewards += reward + 10
                time_rewards += time_reward
                distance_rewards += distance_reward
            new_experiences['obs_side'].update(new_experience)
            self.update_env_vc_info(vc, self._step, road, next_state[-1], done=True, last_reward=True)
            self.success_routing(vc)

        if self.record:
            if self._step % 10 == 0:
                self.recorder.flush()

        if self._step >= self.step_count:
            for vc, _ in self.vehicles.items():
                if self.vehicles[vc]['is_new']:
                    self.success_routing(vc, timeout=True)
                    pass
                elif self.vehicles[vc]['first_decision'] and self.vehicles[vc]['last_reward'] is False:
                    new_experience = {vc: {'next_state': None, 'available_action': None, 'reward': [0], 'success': 0, 'timeout': 1, 'action_signal': 0}}
                    new_experiences['obs_side'].update(new_experience)
                    self.success_routing(vc, timeout=True)
                    pass
                else:
                    if self.vehicles[vc]['last_reward'] is not True:
                        distance_reward = self.extra_reward(vc, self.vehicles[vc]['last_road'], self.vehicles[vc]['next_road'])
                        time_reward = -(self._step-self.vehicles[vc]['time']) / 100
                        if self.reward == 'only_distance':
                            r = distance_reward
                        elif self.reward == 'distance':
                            r = time_reward + distance_reward*self.balancing_coef
                        else:
                            r = time_reward
                        reward = [r, -10]
                        new_experience = {vc: {'next_state': None, 'available_action': None, 'reward': reward, 'success': 0, 'timeout': 1, 'action_signal': 0,\
                                       'next_adj_state': None, 'next_adj_road_neighbor_state': None, 'next_adj_road_neighbor_mask': None, \
                                        'next_adj_road_neighbor_dest_angle': None, 'next_adj_road_neighbor_past_angle': None, \
                                            'next_corr_state': None, 'next_corr_road_neighbor_state': None, 'next_corr_road_neighbor_mask': None, \
                                                'next_corr_road_neighbor_dest_angle': None, 'next_corr_road_neighbor_past_angle': None}}
                        new_experiences['obs_side'].update(new_experience)
                        all_rewards += r - 10
                        time_rewards += time_reward
                        distance_rewards += distance_reward
                        self.success_routing(vc, timeout=True)

            self.info['ATT_success'] = self.success_travel_time/self.success_travel
            self.info['ATT'] = self.total_travel_time/self.total_travel
            self.info['Throughput'] = self.success_travel
            self.info['VEH'] = self.total_travel
            self._step = 0
            self.reset()

            if self.record:
                exit()
        
        self.info['rewards'] = all_rewards
        self.info['time_reward'] = time_rewards
        self.info['distance_reward'] = distance_rewards
        info = self.info
        return new_experiences, next_veh, success_veh, info, next_states, next_acs, next_adj_states, next_adj_road_neighbor_states, next_adj_road_neighbor_masks, next_adj_road_neighbor_dest_angles, next_adj_road_neighbor_past_angles, next_corr_states, next_corr_road_neighbor_states, next_corr_road_neighbor_masks, next_corr_road_neighbor_dest_angles, next_corr_road_neighbor_past_angles
    
def save(model, save_dir, type='present'):
    torch.save(model.state_dict(), str(save_dir) + "/{}.pt".format(type))


class Replay_tmp:
    def __init__(self, max_size):
        self.ob = deque([], max_size)
        self.action = deque([], max_size)
        self.available_action = deque([], max_size)
        self.reward = deque([], max_size)
        self.done = deque([], max_size)
        self.adj_state = deque([], max_size)
        self.adj_road_neighbor_state = deque([], max_size)
        self.adj_road_neighbor_mask = deque([], max_size)
        self.adj_road_neighbor_dest_angle = deque([], max_size)
        self.adj_road_neighbor_past_angle = deque([], max_size)
        self.corr_state = deque([], max_size)
        self.corr_road_neighbor_state = deque([], max_size)
        self.corr_road_neighbor_mask = deque([], max_size)      
        self.corr_road_neighbor_dest_angle = deque([], max_size)
        self.corr_road_neighbor_past_angle = deque([], max_size)

    def pop(self):
        ob = self.ob.popleft()
        action = self.action.popleft()
        available_action = self.available_action.popleft()
        reward = self.reward.popleft()
        done = self.done.popleft()
        adj_state = self.adj_state.popleft()
        adj_road_neighbor_state = self.adj_road_neighbor_state.popleft()
        adj_road_neighbor_mask = self.adj_road_neighbor_mask.popleft()
        adj_road_neighbor_dest_angle = self.adj_road_neighbor_dest_angle.popleft()
        adj_road_neighbor_past_angle = self.adj_road_neighbor_past_angle.popleft()
        corr_state = self.corr_state.popleft()
        corr_road_neighbor_state = self.corr_road_neighbor_state.popleft()
        corr_road_neighbor_mask = self.corr_road_neighbor_mask.popleft()
        corr_road_neighbor_dest_angle = self.corr_road_neighbor_dest_angle.popleft()
        corr_road_neighbor_past_angle = self.corr_road_neighbor_past_angle.popleft()
        next_ob = self.ob[0]
        next_adj_state = self.adj_state[0]
        next_adj_road_neighbor_state = self.adj_road_neighbor_state[0]
        next_adj_road_neighbor_mask = self.adj_road_neighbor_mask[0]
        next_adj_road_neighbor_dest_angle = self.adj_road_neighbor_dest_angle[0]
        next_adj_road_neighbor_past_angle = self.adj_road_neighbor_past_angle[0]
        next_corr_state = self.corr_state[0]
        next_corr_road_neighbor_state = self.corr_road_neighbor_state[0]
        next_corr_road_neighbor_mask = self.corr_road_neighbor_mask[0]
        next_corr_road_neighbor_dest_angle = self.corr_road_neighbor_dest_angle[0]
        next_corr_road_neighbor_past_angle = self.corr_road_neighbor_past_angle[0]
        return [ob, action, available_action, reward, done, next_ob, adj_state, next_adj_state, adj_road_neighbor_state, next_adj_road_neighbor_state, \
                adj_road_neighbor_mask, next_adj_road_neighbor_mask, adj_road_neighbor_dest_angle, next_adj_road_neighbor_dest_angle, \
                    adj_road_neighbor_past_angle, next_adj_road_neighbor_past_angle, corr_state, next_corr_state, corr_road_neighbor_state, next_corr_road_neighbor_state, \
                        corr_road_neighbor_mask, next_corr_road_neighbor_mask, corr_road_neighbor_dest_angle, next_corr_road_neighbor_dest_angle, corr_road_neighbor_past_angle, next_corr_road_neighbor_past_angle]


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.replay = deque([], max_size)
        self.replay_tmp = defaultdict(lambda: Replay_tmp(max_size)) 

    def len(self):
        return len(self.replay)

    def add(self, experiences):
        self.replay.extend(experiences)

    def sample(self, batchsize, transpose=False):
        s, a, ac, r, d, sp, adj_state, next_adj_state, adj_road_neighbor_state, next_adj_road_neighbor_state, \
            adj_road_neighbor_mask, next_adj_road_neighbor_mask, adj_road_neighbor_dest_angle, next_adj_road_neighbor_dest_angle, \
                adj_road_neighbor_past_angle, next_adj_road_neighbor_past_angle, corr_state, next_corr_state, corr_road_neighbor_state, next_corr_road_neighbor_state, \
                    corr_road_neighbor_mask, next_corr_road_neighbor_mask, corr_road_neighbor_dest_angle, next_corr_road_neighbor_dest_angle, corr_road_neighbor_past_angle, next_corr_road_neighbor_past_angle = \
                        zip(*random.sample(self.replay, min(len(self.replay), batchsize)))
        if transpose:
            s, a, r, sp, ac, adj_state, next_adj_state, adj_road_neighbor_state, next_adj_road_neighbor_state, \
                adj_road_neighbor_mask, next_adj_road_neighbor_mask, adj_road_neighbor_dest_angle, next_adj_road_neighbor_dest_angle, \
                    adj_road_neighbor_past_angle, next_adj_road_neighbor_past_angle, corr_state, next_corr_state, corr_road_neighbor_state, next_corr_road_neighbor_state, \
                        corr_road_neighbor_mask, next_corr_road_neighbor_mask, corr_road_neighbor_dest_angle, next_corr_road_neighbor_dest_angle, corr_road_neighbor_past_angle, next_corr_road_neighbor_past_angle = \
                         (list(zip(*i)) for i in [s, a, r, sp, ac, adj_state, next_adj_state, adj_road_neighbor_state, next_adj_road_neighbor_state, \
                adj_road_neighbor_mask, next_adj_road_neighbor_mask, adj_road_neighbor_dest_angle, next_adj_road_neighbor_dest_angle, \
                    adj_road_neighbor_past_angle, next_adj_road_neighbor_past_angle, corr_state, next_corr_state, corr_road_neighbor_state, next_corr_road_neighbor_state, \
                        corr_road_neighbor_mask, next_corr_road_neighbor_mask, corr_road_neighbor_dest_angle, next_corr_road_neighbor_dest_angle, corr_road_neighbor_past_angle, next_corr_road_neighbor_past_angle])
        return s, a, r, sp, d, ac, adj_state, next_adj_state, adj_road_neighbor_state, next_adj_road_neighbor_state, \
            adj_road_neighbor_mask, next_adj_road_neighbor_mask, adj_road_neighbor_dest_angle, next_adj_road_neighbor_dest_angle, \
                adj_road_neighbor_past_angle, next_adj_road_neighbor_past_angle, corr_state, next_corr_state, corr_road_neighbor_state, next_corr_road_neighbor_state, \
                    corr_road_neighbor_mask, next_corr_road_neighbor_mask, corr_road_neighbor_dest_angle, next_corr_road_neighbor_dest_angle, corr_road_neighbor_past_angle, next_corr_road_neighbor_past_angle

    def add_tmp(self, experiences):     ### 给每个临时experience加观察
        timeout = 0
        for veh in experiences['action_side']:
            if experiences['action_side'][veh]['action_signal'] == 1:
                self.replay_tmp[veh].action.append(experiences['action_side'][veh]['action'])
                assert len(self.replay_tmp[veh].action) == len(self.replay_tmp[veh].ob)
        
        for veh in experiences['obs_side']:
            if experiences['obs_side'][veh]['success'] != 1 and experiences['obs_side'][veh]['timeout'] != 1:
                self.replay_tmp[veh].ob.append(experiences['obs_side'][veh]['next_state'])
                self.replay_tmp[veh].available_action.append(experiences['obs_side'][veh]['available_action'])
                self.replay_tmp[veh].done.append(0)
                if experiences['obs_side'][veh]['reward'] is not None:
                    self.replay_tmp[veh].reward.append(experiences['obs_side'][veh]['reward'])
                self.replay_tmp[veh].adj_state.append(experiences['obs_side'][veh]['next_adj_state'])
                self.replay_tmp[veh].adj_road_neighbor_state.append(experiences['obs_side'][veh]['next_adj_road_neighbor_state'])
                self.replay_tmp[veh].adj_road_neighbor_mask.append(experiences['obs_side'][veh]['next_adj_road_neighbor_mask'])
                self.replay_tmp[veh].adj_road_neighbor_dest_angle.append(experiences['obs_side'][veh]['next_adj_road_neighbor_dest_angle'])
                self.replay_tmp[veh].adj_road_neighbor_past_angle.append(experiences['obs_side'][veh]['next_adj_road_neighbor_past_angle'])
                self.replay_tmp[veh].corr_state.append(experiences['obs_side'][veh]['next_corr_state'])
                self.replay_tmp[veh].corr_road_neighbor_state.append(experiences['obs_side'][veh]['next_corr_road_neighbor_state'])
                self.replay_tmp[veh].corr_road_neighbor_mask.append(experiences['obs_side'][veh]['next_corr_road_neighbor_mask'])
                self.replay_tmp[veh].corr_road_neighbor_dest_angle.append(experiences['obs_side'][veh]['next_corr_road_neighbor_dest_angle'])
                self.replay_tmp[veh].corr_road_neighbor_past_angle.append(experiences['obs_side'][veh]['next_corr_road_neighbor_past_angle'])
            elif experiences['obs_side'][veh]['success'] == 1:
                self.replay_tmp[veh].ob.append(experiences['obs_side'][veh]['next_state'])
                self.replay_tmp[veh].done[-1] = 1
                for reward in experiences['obs_side'][veh]['reward']:
                    self.replay_tmp[veh].reward.append(reward)
                self.replay_tmp[veh].adj_state.append(experiences['obs_side'][veh]['next_adj_state'])
                self.replay_tmp[veh].adj_road_neighbor_state.append(experiences['obs_side'][veh]['next_adj_road_neighbor_state'])
                self.replay_tmp[veh].adj_road_neighbor_mask.append(experiences['obs_side'][veh]['next_adj_road_neighbor_mask'])
                self.replay_tmp[veh].adj_road_neighbor_dest_angle.append(experiences['obs_side'][veh]['next_adj_road_neighbor_dest_angle'])
                self.replay_tmp[veh].adj_road_neighbor_past_angle.append(experiences['obs_side'][veh]['next_adj_road_neighbor_past_angle'])
                self.replay_tmp[veh].corr_state.append(experiences['obs_side'][veh]['next_corr_state'])
                self.replay_tmp[veh].corr_road_neighbor_state.append(experiences['obs_side'][veh]['next_corr_road_neighbor_state'])
                self.replay_tmp[veh].corr_road_neighbor_mask.append(experiences['obs_side'][veh]['next_corr_road_neighbor_mask'])
                self.replay_tmp[veh].corr_road_neighbor_dest_angle.append(experiences['obs_side'][veh]['next_corr_road_neighbor_dest_angle'])
                self.replay_tmp[veh].corr_road_neighbor_past_angle.append(experiences['obs_side'][veh]['next_corr_road_neighbor_past_angle'])
            else:
                timeout = 1
                self.replay_tmp[veh].ob.append(np.full_like(self.replay_tmp[veh].ob[-1], -1))
                self.replay_tmp[veh].adj_state.append(np.full_like(self.replay_tmp[veh].adj_state[-1], -1))
                self.replay_tmp[veh].adj_road_neighbor_state.append(np.full_like(self.replay_tmp[veh].adj_road_neighbor_state[-1], -1))
                self.replay_tmp[veh].adj_road_neighbor_mask.append(np.full_like(self.replay_tmp[veh].adj_road_neighbor_mask[-1], -1))
                self.replay_tmp[veh].adj_road_neighbor_dest_angle.append(np.full_like(self.replay_tmp[veh].adj_road_neighbor_dest_angle[-1], -1))
                self.replay_tmp[veh].adj_road_neighbor_past_angle.append(np.full_like(self.replay_tmp[veh].adj_road_neighbor_past_angle[-1], -1))
                self.replay_tmp[veh].corr_state.append(np.full_like(self.replay_tmp[veh].corr_state[-1], -1))
                self.replay_tmp[veh].corr_road_neighbor_state.append(np.full_like(self.replay_tmp[veh].corr_road_neighbor_state[-1], -1))
                self.replay_tmp[veh].corr_road_neighbor_mask.append(np.full_like(self.replay_tmp[veh].corr_road_neighbor_mask[-1], -1)) 
                self.replay_tmp[veh].corr_road_neighbor_dest_angle.append(np.full_like(self.replay_tmp[veh].corr_road_neighbor_dest_angle[-1], -1))
                self.replay_tmp[veh].corr_road_neighbor_past_angle.append(np.full_like(self.replay_tmp[veh].corr_road_neighbor_past_angle[-1], -1))
                for reward in experiences['obs_side'][veh]['reward']:
                    self.replay_tmp[veh].reward.append(reward)
                if len(self.replay_tmp[veh].ob) - 1 == len(self.replay_tmp[veh].reward) == len(self.replay_tmp[veh].done) == len(self.replay_tmp[veh].action) == len(self.replay_tmp[veh].available_action):
                    pass
                else:
                    print(veh, len(self.replay_tmp[veh].reward), len(self.replay_tmp[veh].ob), len(self.replay_tmp[veh].done), len(self.replay_tmp[veh].action), len(self.replay_tmp[veh].available_action))
                self.replay_tmp[veh].done[-1] = 1

        agg_experience = []
        for veh in self.replay_tmp:
            for i in range(len(self.replay_tmp[veh].reward)):
                agg_experience.append(self.replay_tmp[veh].pop())
        
        self.add(agg_experience)

        if timeout:
            self.replay_tmp = defaultdict(lambda: Replay_tmp(self.max_size))        ### 清空临时buffer

        return len(agg_experience)

def lerp(a, b, t):
    t = min(1, t)
    return a*(1-t)+b*t

def main():
    parser = get_config()
    parser.add_argument('--data', type=str, default='data/hangzhou')
    parser.add_argument('--step_count', type=int, default=3600)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--reward', type=str, default='time')   # 是否加入附加reward
    parser.add_argument('--reward_weight', type=float, default=1)
    parser.add_argument('--tl_interval', type=int, default=15, help='interval of tl policies')
    parser.add_argument('--algo', choices=['ft_builtin', 'mp_builtin'], default='mp_builtin')
    parser.add_argument('--training_step', type=int, default=10000000)
    parser.add_argument('--training_start', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--buffer_size', type=int, default=2**20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--mlp', type=str, default='256,256')
    parser.add_argument("--load", type=int, default=0,help='pretrain (default: 0)')
    parser.add_argument("--record", type=int, default=0, help='whether to record the trajectories of vehicles')
    parser.add_argument('--yellow', type=int, default=0, help='yellow time duration')
    parser.add_argument('--save', type=int, default=0, help='save the model')
    parser.add_argument("--layer_N", type=int, default=1, help="Number of layers for actor/critic networks")   
    parser.add_argument("--experience_threshold", type=int, default=300, help="Number of experiences before training")
    parser.add_argument("--update_threshold", type=int, default=10, help="Number of experiences before updating")
    parser.add_argument("--target_vehicle_id", type=int, default=13, help='target vehicle id')
    parser.add_argument("--balancing_coef", type=float, default=0.5, help='balancing coefficient for two rewards')
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--dqn_type", type=str, choices=['dqn', 'dueling'], default='dqn')
    parser.add_argument('--agg_type', type=str, choices=['none', 'sa', 'lc'], default='none')
    parser.add_argument('--lc_interval', type=int, default=300, help='interval of lc policies')
    parser.add_argument('--basic_update_times', type=int, default=1)
    parser.add_argument('--exploration_times', type=int, default=4000000)
    args = parser.parse_args()

    args.city = args.data.split('/')[-1]
    
    setproctitle.setproctitle('Router@zengjinwei')

    path = 'log/router/{}_{}_gamma={}_lr={}_batch_size={}_reward={}_rw={}_ts={}_et={}_ut={}_et={}_{}'.format(args.data, args.dqn_type, args.gamma, args.lr, args.batchsize, args.reward, args.reward_weight, args.training_start, args.experience_threshold, args.update_threshold, args.exploration_times, time.strftime('%d%H%M'))
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/cmd.sh', 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\ntensorboard --port 8888 --logdir '+os.path.abspath(path))
    with open(f'{path}/args.json', 'w') as f:
        json.dump(vars(args), f)
    device = torch.device("cuda")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)

    writer = SummaryWriter(path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    if args.algo == 'ft_builtin':
        base_algo = TlPolicy.FIXED_TIME
    elif args.algo == 'mp_builtin':
        base_algo = TlPolicy.MAX_PRESSURE
    else:
        raise NotImplementedError
    
    if args.agg_type == 'none':
        args.agg = 0
        args.corr_agg = 0
    elif args.agg_type == 'sa':
        args.agg = 1
        args.corr_agg = 0
    elif args.agg_type == 'lc':
        args.agg = 1
        args.corr_agg = 1
    else:
        raise NotImplementedError
    
    env = Env(
        data_path=args.data,
        step_size=args.interval,
        step_count=args.step_count//args.interval,
        log_dir='',
        base_algo=base_algo,
        reward=args.reward,
        tl_interval=args.tl_interval,
        yellow_time=args.yellow,
        save=args.save, 
        record=args.record, 
        args=args,
        reward_weight=args.reward_weight
    )

    args.road2adjroad = env.road2adjroad
    args.vehid2idxs = env.veh_id2idxs
    args.veh_dest_roads = env.veh_dest_roads
    args.persons = env.eng.get_persons(False)
    args.max_action_size = env.max_action_size
    args.agent_ids = env.selected_person_ids

    ### 共享的replay
    replay = ReplayBuffer(args.buffer_size)
    if args.dqn_type == 'dqn':
        Q = R_Actor(args,
                    source_state_dim=env.source_state_dim,
                    neighbor_state_dim=env.neighbor_state_dim,
                    edge_dim=env.edge_dim,
                    max_actions=env.max_action_size, 
                    device=device).to(device)       ### 多个agent共享一个策略网络
    elif args.dqn_type == 'dueling':
        Q = VR_Actor(args,
                    source_state_dim=env.source_state_dim,
                    neighbor_state_dim=env.neighbor_state_dim,
                    edge_dim=env.edge_dim,
                    max_actions=env.max_action_size, 
                    device=device).to(device)
    else:
        raise NotImplementedError
    
    if args.load == 1:
        data_path = 'log/data/grid_network_multi_agent_dqn_gamma=0.995_lr=1e-05_batch_size=1024_reward=distance_rw=1_ts=30000_et=400_ut=30_et=6000000_131524'
        load_dir = str(data_path) + "/best.pt"
        Q.load_state_dict(torch.load(load_dir))

    warmup_steps = 10000
    total_steps = 2*10**6
    final_lr = args.lr
    initial_lr = 5e-3
    def lr_schedule_fn(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return max(1.0 - progress, final_lr / initial_lr)

    opt = torch.optim.Adam(Q.parameters(), lr=initial_lr)
    scheduler = LambdaLR(opt, lr_lambda=lr_schedule_fn)

    Q_target = deepcopy(Q)
    Q_target = Q_target.to(device)
    
    vehs, roads = [], []   ## 初始时没有需要决策的vehs
    success_vehs = []
    episode_reward = 0
    time_reward, distance_reward = 0, 0
    episode_count, episode_step, episode_num = 0, 0, 0
    best_episode_reward = -1e999
    args.steps = 0
    average_travel_time = 0
    average_finish_rate = 0

    basic_batch_size = args.batchsize
    basic_update_times = args.basic_update_times         ### 之前设为5
    replay_max = args.buffer_size

    added_experiences = 0
    training_count = 0

    with tqdm(range(args.training_step), ncols=100, smoothing=0.1) as bar:
        for step in bar:
            t0 = time.time()
            _st = time.time()
            eps = lerp(1, 0.05, step/args.exploration_times)   ### 到2000000步时epsilon为0.05      ### 0408 修改为1000000尝试一下
            actions = []
            if len(vehs) > 0:
                # 需要读取obs和available_actions
                obs = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
                if args.agg == 1:
                    adj_states = torch.tensor(np.array(adj_states), dtype=torch.float32, device=device)
                    adj_neighbor_states = torch.tensor(np.array(adj_neighbor_states), dtype=torch.float32, device=device)
                    adj_neighbor_masks = torch.tensor(np.array(adj_neighbor_masks), dtype=torch.float32, device=device)
                    adj_neighbor_dest_angles = torch.tensor(np.array(adj_neighbor_dest_angles), dtype=torch.float32, device=device)
                    adj_neighbor_past_angles = torch.tensor(np.array(adj_neighbor_past_angles), dtype=torch.float32, device=device)
                if args.corr_agg == 1:
                    corr_states = torch.tensor(np.array(corr_states), dtype=torch.float32, device=device)
                    corr_neighbor_states = torch.tensor(np.array(corr_neighbor_states), dtype=torch.float32, device=device)
                    corr_neighbor_masks = torch.tensor(np.array(corr_neighbor_masks), dtype=torch.float32, device=device)
                    corr_neighbor_dest_angles = torch.tensor(np.array(corr_neighbor_dest_angles), dtype=torch.float32, device=device)
                    corr_neighbor_past_angles = torch.tensor(np.array(corr_neighbor_past_angles), dtype=torch.float32, device=device)    
                available_actions = torch.tensor(np.array(available_actions), dtype=torch.float32, device=device)
                ac = available_actions.sum(axis=1)

                action_explore = [random.randint(0, a-1) for a in ac]
                if step < args.training_start and args.load == 0:
                    actions = action_explore
                else:
                    with torch.no_grad():
                        m = Q(obs, adj_states, adj_neighbor_states, adj_neighbor_masks, adj_neighbor_dest_angles, adj_neighbor_past_angles, corr_states, corr_neighbor_states, corr_neighbor_masks, corr_neighbor_dest_angles, corr_neighbor_past_angles)
                        m[available_actions==0] = -1e9
                        action_exploit = torch.argmax(m, dim=-1).cpu().numpy()
                    if args.load == 0:
                        actions = np.choose(np.random.uniform(size=len(ac)) < eps, [action_explore, action_exploit])
                    else:
                        actions = action_exploit

            new_experiences, next_vehs, success_vehs, infos, next_states, next_acs, next_adj_states, next_adj_neighbor_states, next_adj_neighbor_masks, next_adj_neighbor_dest_angles, next_adj_neighbor_past_angles, next_corr_states, next_corr_neighbor_states, next_corr_neighbor_masks, next_corr_neighbor_dest_angles, next_corr_neighbor_past_angles = env.step(actions)
            added_experiences += replay.add_tmp(new_experiences)
             
            episode_reward += infos['rewards']
            time_reward += infos['time_reward']
            distance_reward += infos['distance_reward']

            episode_count += 1
            episode_step += 1
            if step >= args.training_start and added_experiences >  args.experience_threshold:
                replay_len = replay.len()
                k = 1 + replay_len / replay_max

                batch_size   = int(k * basic_batch_size)
                update_times = int(k * basic_update_times)
                overall_loss = 0
                for _ in range(update_times):
                    s, a, r, sp, d, ac, adj_state, next_adj_state, adj_road_neighbor_state, next_adj_road_neighbor_state, \
                        adj_road_neighbor_mask, next_adj_road_neighbor_mask, adj_road_neighbor_dest_angle, next_adj_road_neighbor_dest_angle, \
                            adj_road_neighbor_past_angle, next_adj_road_neighbor_past_angle, corr_state, next_corr_state, corr_road_neighbor_state, next_corr_road_neighbor_state, \
                                corr_road_neighbor_mask, next_corr_road_neighbor_mask, corr_road_neighbor_dest_angle, next_corr_road_neighbor_dest_angle, corr_road_neighbor_past_angle, next_corr_road_neighbor_past_angle = \
                                    replay.sample(batch_size, transpose=False)
                    d = torch.tensor(d, dtype=torch.float32, device=device)
                    loss = 0
                    s = torch.tensor(np.array(s), dtype=torch.float32, device=device)
                    a = torch.tensor(a, dtype=torch.long, device=device)
                    r = torch.tensor(r, dtype=torch.float32, device=device)
                    sp = torch.tensor(np.array(sp), dtype=torch.float32, device=device)
                    ac = torch.tensor(ac, dtype=torch.float32, device=device)
                    if args.agg == 1:
                        adj_state = torch.tensor(np.array(adj_state), dtype=torch.float32, device=device)
                        next_adj_state = torch.tensor(np.array(next_adj_state), dtype=torch.float32, device=device)
                        adj_road_neighbor_state = torch.tensor(np.array(adj_road_neighbor_state), dtype=torch.float32, device=device)
                        next_adj_road_neighbor_state = torch.tensor(np.array(next_adj_road_neighbor_state), dtype=torch.float32, device=device)
                        adj_road_neighbor_mask = torch.tensor(np.array(adj_road_neighbor_mask), dtype=torch.float32, device=device)
                        next_adj_road_neighbor_mask = torch.tensor(np.array(next_adj_road_neighbor_mask), dtype=torch.float32, device=device)
                        adj_road_neighbor_dest_angle = torch.tensor(np.array(adj_road_neighbor_dest_angle), dtype=torch.float32, device=device)
                        next_adj_road_neighbor_dest_angle = torch.tensor(np.array(next_adj_road_neighbor_dest_angle), dtype=torch.float32, device=device)
                        adj_road_neighbor_past_angle = torch.tensor(np.array(adj_road_neighbor_past_angle), dtype=torch.float32, device=device)
                        next_adj_road_neighbor_past_angle = torch.tensor(np.array(next_adj_road_neighbor_past_angle), dtype=torch.float32, device=device)
                    if args.corr_agg == 1:
                        corr_state = torch.tensor(np.array(corr_state), dtype=torch.float32, device=device)
                        next_corr_state = torch.tensor(np.array(next_corr_state), dtype=torch.float32, device=device)
                        corr_road_neighbor_state = torch.tensor(np.array(corr_road_neighbor_state), dtype=torch.float32, device=device)
                        next_corr_road_neighbor_state = torch.tensor(np.array(next_corr_road_neighbor_state), dtype=torch.float32, device=device)
                        corr_road_neighbor_mask = torch.tensor(np.array(corr_road_neighbor_mask), dtype=torch.float32, device=device)
                        next_corr_road_neighbor_mask = torch.tensor(np.array(next_corr_road_neighbor_mask), dtype=torch.float32, device=device)
                        corr_road_neighbor_dest_angle = torch.tensor(np.array(corr_road_neighbor_dest_angle), dtype=torch.float32, device=device)
                        next_corr_road_neighbor_dest_angle = torch.tensor(np.array(next_corr_road_neighbor_dest_angle), dtype=torch.float32, device=device)
                        corr_road_neighbor_past_angle = torch.tensor(np.array(corr_road_neighbor_past_angle), dtype=torch.float32, device=device)
                        next_corr_road_neighbor_past_angle = torch.tensor(np.array(next_corr_road_neighbor_past_angle), dtype=torch.float32, device=device)
                    with torch.no_grad():
                        m = Q_target(sp, next_adj_state, next_adj_road_neighbor_state, next_adj_road_neighbor_mask, next_adj_road_neighbor_dest_angle, next_adj_road_neighbor_past_angle, next_corr_state, next_corr_road_neighbor_state, next_corr_road_neighbor_mask, next_corr_road_neighbor_dest_angle, next_corr_road_neighbor_past_angle)
                        m[ac==0] = -1e9
                        y_target = r+args.gamma*m.max(1).values*(1-d)
                    y = Q(s, adj_state, adj_road_neighbor_state, adj_road_neighbor_mask, adj_road_neighbor_dest_angle, adj_road_neighbor_past_angle, corr_state, corr_road_neighbor_state, corr_road_neighbor_mask, corr_road_neighbor_dest_angle, corr_road_neighbor_past_angle).gather(-1, a[..., None]).view(-1)
                    loss = loss+F.mse_loss(y, y_target)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    scheduler.step()
                    overall_loss += loss.item()
                overall_loss /= update_times
                writer.add_scalar('metric/overall_loss', overall_loss, step)
                added_experiences = 0
                training_count += 1
                if training_count % args.update_threshold == 0:
                    Q_target.load_state_dict(Q.state_dict())
                    training_count = 0

            vehs = next_vehs
            obs = next_states
            adj_states = next_adj_states
            adj_neighbor_states = next_adj_neighbor_states
            adj_neighbor_masks = next_adj_neighbor_masks
            adj_neighbor_dest_angles = next_adj_neighbor_dest_angles
            adj_neighbor_past_angles = next_adj_neighbor_past_angles
            corr_states = next_corr_states
            corr_neighbor_states = next_corr_neighbor_states
            corr_neighbor_masks = next_corr_neighbor_masks
            corr_neighbor_dest_angles = next_corr_neighbor_dest_angles
            corr_neighbor_past_angles = next_corr_neighbor_past_angles
            available_actions = next_acs

            args.steps += 1

            if args.steps//args.interval % args.step_count == 0:
                if args.first == 1:
                    print(replay.len())
                    exit()
                writer.add_scalar('metric/EpisodeReward', episode_reward, episode_num)
                writer.add_scalar('metric/time_reward', time_reward, episode_num)
                writer.add_scalar('metric/distance_reward', distance_reward, episode_num)
                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    writer.add_scalar('metric/Best_EpisodeReward', episode_reward)      
                    writer.add_scalar('metric/Best_ATT', infos['ATT'])
                    writer.add_scalar('metric/Best_ATT_finished', infos['ATT_success'])
                    writer.add_scalar('metric/Best_Throughput', infos['Throughput'])
                    writer.add_scalar('metric/Best_VEH', infos['VEH'])
                    save(Q, path, 'best')
                episode_num += 1
                episode_reward = 0
                time_reward, distance_reward = 0, 0
                episode_count = 0
                writer.add_scalar('metric/ATT', infos['ATT'], step)
                writer.add_scalar('metric/ATT_finished', infos['ATT_success'], step)
                writer.add_scalar('metric/Throughput', infos['Throughput'], step)
                writer.add_scalar('metric/VEH', infos['VEH'], step)
                args.steps = 0
                vehs = []
                obs = []
                available_actions = []
                bar.set_description(f'Step: {step}, ATT: {infos["ATT"]:.2f}, TP: {infos["Throughput"]}')

                all_rewards = infos['rewards']
                writer.add_scalar('metric/Reward', all_rewards, step)            
                writer.add_scalar('chart/FPS', 1/(time.time()-_st), step)
                episode_step = 0
  
                if args.load == 1:
                    exit()


if __name__ == '__main__':
    main()
