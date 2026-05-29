import copy
import math
import os
import pickle
import time
import random
import warnings

import dgl
import torch
import pandas as pd
from bintrees import AVLTree
from scipy.spatial import cKDTree
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import geo_math
from config import CONFIG


class POIBase:
    def __init__(self, poi_path, neighbor_radius=100):
        """
        Initialize POI knowledge graph.
        pois: list of POI ids
        relations: list of tuples (src_id, dst_id, relation_type)
        """
        self.kd_tree = None
        self.pois_df = None
        self.geo_math = geo_math()
        self.load_poi(poi_path)  # load poi data
        self.bbox = self.get_boundary_box()
        print(f"boundary box: {self.bbox}")
        
        self.neighbor_radius = neighbor_radius  # 100 meters
        self.space_radius = self.geo_math.arc_length_to_chord_length(self.neighbor_radius)
    
    def get_boundary_box(self):
        min_lon, max_lon, min_lat, max_lat = float('inf'), float('-inf'), float('inf'), float('-inf')
        for idx, poi in self.pois_df.iterrows():
            lon, lat = poi["lon"], poi["lat"]
            min_lon = min(min_lon, lon)
            max_lon = max(max_lon, lon)
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)
        return (min_lon, max_lon, min_lat, max_lat)

    def load_poi(self, path):
        dtype_spec = {
            'id': 'int32',
            'str_id': 'str',
            'poi_name': 'str',
            'category': 'str',
            'interest': 'str',
            'city': 'str',
            'lon': 'float64',
            'lat': 'float64',
        }
        self.pois_df = pd.read_csv(path, dtype=dtype_spec)
        self.pois_df['category_id'], self.cate_list = pd.factorize(self.pois_df['category'] + self.pois_df['interest'])
        self.get_kd_tree()

    def get_kd_tree(self):
        coord_3d = [self.geo_math.llh_to_ecef(row["lon"], row["lat"]) for i, row in self.pois_df.iterrows()]  # Unit: meters
        self.kd_tree = cKDTree(coord_3d)
        print("[NOTICE] constructed kd tree")

    def query_neighbor(self, poi_id):  # what are the spatial neighbors of a POI
        coord_3d = self.geo_math.llh_to_ecef(self.pois_df.iloc[poi_id]["lon"], self.pois_df.iloc[poi_id]["lat"])
        neighbors = self.kd_tree.query_ball_point(coord_3d, self.space_radius)
        neighbors.remove(poi_id)
        return neighbors

    def query_types(self, poi_id, target_set):  # what are the POIs of the same type
        source_row = self.pois_df.iloc[poi_id]
        target_df = self.pois_df.iloc[target_set]
        sim_df = target_df[(target_df["category"] == source_row["category"]) &
                           (target_df["interest"] == source_row["interest"]) & (target_df["id"] != poi_id)]
        similar = sim_df["id"].tolist()
        # if len(similar)>50:
        #     print("blocked")
        return similar

    def get_poi_info(self, poi_ids):
        return self.pois_df.iloc[poi_ids]


class TrajDataset(Dataset):
    def __init__(self, pp_path, action, shuffle=False, data_index=None):
        print(f"initiating {action} dataset...")
        self.pp_path = pp_path
        self.dataset = self._load_data()  # load data
        self.action = action
        self.action_mark = 0 if self.action == "train" else 1 if self.action == "valid" else 2
        self.data_order = self.get_data_order()  # get all the valid data records
        self.data_index = list(range(len(self.data_order)))  # get a visiting list
        if shuffle and data_index is None:
            self._shuffle_index()
        elif data_index is not None:
            self.data_index = copy.deepcopy(data_index)
            print("[CONFIG] Use external data_index with {} records.".format(len(self.data_index)))

    def get_data_order(self, data_order=None):
        order = []
        for user in self.dataset:
            tvt = self._tvt_split(user)
            start, end = tvt[self.action_mark-1], tvt[self.action_mark]
            if tvt[self.action_mark-1] < tvt[self.action_mark]:
                order += [{"index": user, "record": record} for record in range(start, end)]
        return order

    def _shuffle_index(self):
        if CONFIG.random_seed != -1:  # shuffle dataset with a seed
            shuffle_seed = CONFIG.random_seed
        else:
            shuffle_seed = random.randint(0, 999999)
        random.seed(shuffle_seed)
        print("[CONFIG] Shuffle dataset with RANDOM seed: {}".format(shuffle_seed))
        random.shuffle(self.data_index)

    def _load_data(self):
        with open(self.pp_path, 'rb') as file:
            start_time = time.time()
            dataset = pickle.load(file)
            print("loaded {} with {} seconds".format(self.pp_path, time.time() - start_time))
        return dataset

    def _tvt_split(self, user):  # small tool, return record ptr scope
        return self.dataset[user]["tvt_split"] + [0]

    def __len__(self):
        return len(self.data_order)

    def __getitem__(self, idx):
        proj_idx = self.data_index[idx]
        record = self.data_order[proj_idx]
        traj_entity = TrajData(self.dataset[record["index"]], record["index"])
        return traj_entity.get_data(record["record"])


class TrajData:
    def __init__(self, user_record, user_id):
        self.user_id = user_id
        self.user_record = user_record
        self.graph = dgl.heterograph({
            ('poi', 'u_trans', 'poi'): (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32)),
            ('poi', 'o_trans', 'poi'): (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32)),
            ('poi', 'near', 'poi'): (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32)),
            ('poi', 'same', 'poi'): (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32)),
            ('poi', 'recommend', 'poi'): (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32)),
        }, idtype=torch.int32)

    def _prepare_type(self, data):
        ret = []
        for i, (a, b) in enumerate(data):
            if isinstance(a, list):
                a, b = torch.tensor(a), torch.tensor(b)
            ret.append((a.type(torch.int32), b.type(torch.int32)))
        return ret[0], ret[1], ret[2], ret[3]

    def _curtail_graph(self, main_nodes):  # curtail graph for the latest n traj nodes
        if self.graph.num_nodes() < CONFIG.curtail_graph:  # if this condition there's no need to curtail
            all_nodes = list(self.graph.nodes())
        else:
            n_o, n_n = set(), set()
            for node in main_nodes:
                n_o = n_o.union(set(self.graph.successors(node, etype='o_trans').tolist()))
                n_n = n_n.union(set(self.graph.successors(node, etype='near').tolist()))
            n_main = set(main_nodes)
            all_nodes = list(n_main.union(n_o).union(n_n))
        nodes_dict = {
            'poi': all_nodes
        }
        subgraph = dgl.node_subgraph(self.graph, nodes_dict)
        return subgraph.clone(), all_nodes

    def _next_graph(self, index):
        (u_src, u_dst), (trans_src, trans_dst), \
        (n_src, n_dst), (s_src, s_dst) = self._prepare_type(self.user_record["graph_split"][index])
        self.graph.add_edges(u_src, u_dst, etype='u_trans')
        self.graph.add_edges(trans_src, trans_dst, etype='o_trans')
        self.graph.add_edges(n_src, n_dst, etype='near')
        self.graph.add_edges(s_src, s_dst, etype='same')
        return self.graph

    def get_data(self, idx):
        for i in range(idx+1):  # prepare graph plus one because idx is the current index, and range(idx+1) is 0~idx 
            self._next_graph(i)

        last_traj_ptr = self.user_record["traj_split"][idx-1] if idx > 0 else 0
        traj_ptr = self.user_record["traj_split"][idx]
        graph_traj_id = [self.user_record["graph_nodes"].index(poi) for poi in
                         self.user_record["traj"][:traj_ptr]]
        try:
            if CONFIG.curtail_graph < len(graph_traj_id):  # if the graph is too large
                graph, all_nodes = self._curtail_graph(graph_traj_id[-CONFIG.curtail_graph:])
            else:
                graph, all_nodes = self.graph, list(self.graph.nodes())
        except:
            print(f"error in curtailing graph at index {idx}")
        # graph, all_nodes = self.graph, list(self.graph.nodes())
        graph_nodes = torch.tensor([self.user_record["graph_nodes"][i] for i in all_nodes])

        traj = self.user_record["traj"][:traj_ptr]
        time_stamps = self.user_record["time_stamps"][:traj_ptr]
        next_visit = self.user_record["traj"][traj_ptr]
        next_time_stamp = self.user_record["time_stamps"][traj_ptr]

        # return self.user_id, graph, graph_nodes, traj, time_stamps, next_visit, next_time_stamp
        return self.user_id, graph, graph_nodes, traj, time_stamps, next_visit, next_time_stamp, last_traj_ptr

