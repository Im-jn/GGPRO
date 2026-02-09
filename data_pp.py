import argparse
import csv
import sys
import time
import os

from nltk import word_tokenize, pos_tag
from collections import deque
from tqdm import tqdm
import random
from bintrees import AVLTree
import pandas as pd
import pickle
import dgl
import torch

from database import POIBase
from config import CONFIG, init
import re

load_step_1 = True
POI_TRANS_RECENT = 20


def contains_letters(s):  # Search if the string contains any English letters
    return re.search(r'[a-zA-Z]', s) is not None


########################################################################
# Trajectory data:
# in this version, we consider an accepted gap for a trajectory and
# construct the trajectory as a whole, instead of updating one node at a time.
########################################################################
def split_text(s, node):
    result = []
    in_quotes = False
    start = 0
    for i, c in enumerate(s):
        if c == '"':
            in_quotes = not in_quotes
        elif c == node and not in_quotes:
            result.append(s[start:i])
            start = i + 1
    result.append(s[start:])
    return result


def make_poi_name(poi_weird_name, city):
    split_name = split_text(poi_weird_name, "-")
    cap_list = [word.capitalize() for word in split_name]
    name = " ".join(cap_list)
    city = " ".join([word.capitalize() for word in split_text(city, " ")])
    if city in name:
        name = name.replace(" "+city, "")

    if len(cap_list) > 5:
        tokens = word_tokenize(name)
        tagged_words = pos_tag(tokens)
        for i in range(len(tagged_words)-1, 0, -1):
            if 'NN' in tagged_words[i][1]:
                name = tagged_words[i-1][0] + " " + tagged_words[i][0]

    return name


def processPOIandUsers(dataframe):
    df = dataframe

    # initialize data structures
    poi_info_store = []
    poi_avl = AVLTree()
    user_info_store = []
    user_avl = AVLTree()

    print("Going through all the POI and User information.")
    for index, row in tqdm(df.iterrows()):
        user_str_id, poi_str_id, t, lat, lon, cate = row['userid'], row['placeid'], row['datetime'], \
                                                           row['lat'], row['lon'], row['category']
        try:
            city = row['city']
            if pd.isna(city):
                city = "Not Specified"
            else:
                city = city.replace(", ", " ")
                city = city.replace(",", " ")
                city = city if contains_letters(city) else "Not Specified"

            poi_name = make_poi_name(poi_str_id, city)
        except:
            city = "Not Specified"
            poi_name = "Not Specified"

        cate = cate.replace(",", "/")
        try:
            category = cate.split(":")[-2]
            interest = cate.split(":")[-1]
        except:
            category = cate
            interest = "Not Specified"

        if poi_avl.get(poi_str_id) is None:
            poi_info_store.append([len(poi_avl), poi_str_id, poi_name, category, interest, city, lon, lat, 1])
            poi_avl.insert(poi_str_id, len(poi_avl))
        else:
            poi_info_store[poi_avl.get(poi_str_id)][-1] += 1

        if user_avl.get(user_str_id) is None:
            user_info_store.append([len(user_avl), user_str_id, 1])
            user_avl.insert(user_str_id, len(user_avl))
        else:
            user_info_store[user_avl.get(user_str_id)][-1] += 1

    print("get {} pois with {} users".format(len(poi_info_store), len(user_info_store)))

    poi_avl = AVLTree()
    user_avl = AVLTree()
    poi_info_list = []
    user_info_list = []
    wastes_poi = []
    wastes_user = []
    print("Filter POIs and Users that are less than {}.".format(CONFIG.min_count))
    for i, record in enumerate(poi_info_store):
        if record[-1] >= CONFIG.min_count:
            record[0] = len(poi_avl)
            poi_info_list.append(record[:-1])
            poi_avl.insert(record[1], len(poi_avl))  # reconstruct the avl tree
        else:
            wastes_poi.append(record[1])
    for i, record in enumerate(user_info_store):
        if record[-1] >= CONFIG.min_count:
            record[0] = len(user_avl)
            user_info_list.append(record[:-1])
            user_avl.insert(record[1], len(user_avl))  # reconstruct the avl tree
        else:
            wastes_user.append(record[1])

    print("remain {} pois with {} users".format(len(poi_info_list), len(user_info_list)))

    return poi_info_list, user_info_list, poi_avl, user_avl, wastes_poi, wastes_user


def switchPlace(count_list, place):
    for i in range(place, 0, -1):
        if count_list[i][0] < count_list[i-1][0]:
            return i
    return 0


def constructDatabase(dataframe, poi_avl, user_avl, poi_base):
    df = dataframe

    poi_trans_count = [{"avl": AVLTree(), "sorted": [], "recent": deque()} for _ in range(len(poi_avl))]
    user_last_visit = [-1 for _ in range(len(user_avl))]
    user_current_visit = [-1 for _ in range(len(user_avl))]

    # construct transition count
    print("Going through all information to construct database.")
    user_database = {}
    total_records = len(df)
    mode = "train"
    for index, (line, row) in tqdm(enumerate(df.iterrows())):
        # switch data action
        if index == int(CONFIG.tvt_rate[0] * total_records):
            print("read {} records, {} total records as train set".format(index, CONFIG.tvt_rate[0]))
            mode = "valid"
        elif index == int((CONFIG.tvt_rate[0] + CONFIG.tvt_rate[1]) * total_records):
            print("read {} records, {} total records as valid set".format(index, CONFIG.tvt_rate[1]))
            mode = "test"

        user_str_id, poi_str_id, t = row['userid'], row['placeid'], row['datetime']
        user_id = user_avl.get(user_str_id)
        next_poi = poi_avl.get(poi_str_id)

        # update user database
        if user_id not in user_database:
            history = {"graph": None, "graph_nodes": [], "nodes_avl": AVLTree(), "graph_split": [],
                       "traj": [], "traj_split": [], "tvt_split": [0, 0, 0], "time_stamps": []}
        else:
            history = user_database[user_id]
        
        if len(history["time_stamps"]) > 0 and history["time_stamps"][-1] == t:
            continue # skip if the time stamp is the same as the last time stamp

        # maintain the transition count
        last_poi = user_last_visit[user_id]
        cur_poi = user_current_visit[user_id]
        if last_poi != -1 and last_poi != cur_poi:  # if there is a transition
            recent_trans = poi_trans_count[last_poi]["recent"]
            recent_trans.append(cur_poi)
            if len(recent_trans) > POI_TRANS_RECENT:
                recent_trans.popleft()

            # rebuild counts from recent transitions to keep only the latest window
            counts = {}
            for poi in recent_trans:
                counts[poi] = counts.get(poi, 0) + 1

            sorted_list = sorted(
                ((count, poi) for poi, count in counts.items()),
                key=lambda item: -item[0]
            )
            avl = AVLTree()
            for idx, (_, poi) in enumerate(sorted_list):
                avl.insert(poi, idx)

            poi_trans_count[last_poi]["sorted"] = sorted_list
            poi_trans_count[last_poi]["avl"] = avl

        user_last_visit[user_id] = cur_poi  # record the last visit
        user_current_visit[user_id] = next_poi  # record the current visit

        if len(history["traj"]) > 0 and checkMinLength(history["traj_split"]):
            # if trajectory is too short, then don't say anything, lets append (except for the change of user id)
            update = constructGraph(history, next_poi, poi_trans_count, poi_base, mode="append", tvt_mode=mode)
        elif (history["tvt_split"][0] == 0 and mode == "train") or \
             (history["tvt_split"][0] == history["tvt_split"][1] and mode == "valid") or \
             (history["tvt_split"][1] == history["tvt_split"][2] and mode == "test"):
            # if dataset switched action
            update = constructGraph(history, next_poi, poi_trans_count, poi_base, mode="new", tvt_mode=mode)
        elif checkIllegalGap(history["time_stamps"], t):
            # if time gap isn't acceptable and the traj isn't too short
            update = constructGraph(history, next_poi, poi_trans_count, poi_base, mode="new", tvt_mode=mode)
        else:
            update = constructGraph(history, next_poi, poi_trans_count, poi_base, mode="append", tvt_mode=mode)
        update["time_stamps"].append(t)

        # renew the data sample's "train-valid-test" split
        samples_count = len(history["traj_split"])
        if mode == "train":
            update["tvt_split"] = [samples_count, samples_count, samples_count]
        elif mode == "valid":
            update["tvt_split"][1] = samples_count
            update["tvt_split"][2] = samples_count
        else:  # test
            update["tvt_split"][2] = samples_count

        user_database[user_id] = update

    return user_database


def checkMinLength(traj_split):
    if len(traj_split) == 0:
        return False
    if traj_split[-1] <= 5:
        return True
    if len(traj_split) > 1 and traj_split[-1] - traj_split[-2] <= 5:
        return True
    return False


def checkIllegalGap(time_list, cur_time):
    if len(time_list) == 0 or (cur_time - time_list[-1]).total_seconds() > 86400:
        return True
    return False


def checkNewEdges(ex_edge_set, new_edge_set):
    unique_edges = new_edge_set - ex_edge_set
    unique_src, unique_dst = zip(*unique_edges) if unique_edges else ([], [])
    return torch.tensor(unique_src, dtype=torch.int32), torch.tensor(unique_dst, dtype=torch.int32)


def constructGraph(history, next_poi, poi_trans_count, poi_base, mode=["new", "append", ""][-1], tvt_mode=None):
    update = history
    if len(history["traj"]) == 0:  # if started
        # start with an empty graph
        update["graph"] = {'u_trans': set(), 'o_trans': set(), 'near': set(), 'same': set()}
        # renew traj
        update["traj"].append(next_poi)  # this is the "next" POI
        # start with an empty graph
        update["graph_split"].append(((torch.tensor([]), torch.tensor([])),
                                      (torch.tensor([]), torch.tensor([])),
                                      (torch.tensor([]), torch.tensor([])),
                                      (torch.tensor([]), torch.tensor([]))))
        update["traj_split"].append(len(update["traj"])-1)  # append a new section
    else:
        cur_poi = update["traj"][-1]
        o_edges, n_edges, s_edges, graph_nodes, nodes_avl = \
            get_new_edges(cur_poi, update["graph_nodes"], update["nodes_avl"], poi_trans_count, poi_base)
        if len(update["traj"]) >= 2:  # if there is a last visited POI
            last_poi = update["traj"][-2]
            u_edge = {(nodes_avl.get(last_poi), nodes_avl.get(cur_poi))}
        else:
            u_edge = set()

        u_src, u_dst = checkNewEdges(update["graph"]['u_trans'], u_edge)
        o_src, o_dst = checkNewEdges(update["graph"]['o_trans'], o_edges)
        n_src, n_dst = checkNewEdges(update["graph"]['near'], n_edges)
        s_src, s_dst = checkNewEdges(update["graph"]['same'], s_edges)

        update["graph"]['u_trans'] = update["graph"]['u_trans'].union(u_edge)
        update["graph"]['o_trans'] = update["graph"]['o_trans'].union(o_edges)
        update["graph"]['near'] = update["graph"]['near'].union(n_edges)
        update["graph"]['same'] = update["graph"]['same'].union(s_edges)

        # renew graph_nodes
        update["graph_nodes"] = graph_nodes
        update["nodes_avl"] = nodes_avl
        # renew traj
        update["traj"].append(next_poi)
        # renew graph split
        if mode == "new":  # this method naturally skipped the first trajectory (since there are no historical traj)
            update["graph_split"].append(((u_src, u_dst), (o_src, o_dst), (n_src, n_dst), (s_src, s_dst)))
            update["traj_split"].append(len(update["traj"])-1)  # append a new section
        elif mode == "append":
            (u1, u2), (o1, o2), (n1, n2), (s1, s2) = update["graph_split"][-1]
            update["graph_split"][-1] = ((torch.cat([u1, u_src]), torch.cat([u2, u_dst])),
                                        (torch.cat([o1, o_src]), torch.cat([o2, o_dst])),
                                        (torch.cat([n1, n_src]), torch.cat([n2, n_dst])),
                                        (torch.cat([s1, s_src]), torch.cat([s2, s_dst])))
            update["traj_split"][-1] = len(update["traj"])-1  # renew the last section
        else:
            raise "No such construct mode"

    return update


# last version of new edges
def get_new_edges(cur_poi, graph_nodes, nodes_avl, poi_trans_count, poi_base):
    trans_edges, n_edges, s_edges = set(), set(), set()
    if nodes_avl.get(cur_poi) is None:  # if the current poi is not in the graph
        cur_index = len(graph_nodes)
        nodes_avl.insert(cur_poi, len(graph_nodes))
        graph_nodes.append(cur_poi)

        # spatial relation
        neighbor = poi_base.query_neighbor(cur_poi)
        for n in neighbor:
            if nodes_avl.get(n) is None:
                nodes_avl.insert(n, len(graph_nodes))
                graph_nodes.append(n)
            n_edges.add((nodes_avl.get(n), cur_index))
            n_edges.add((cur_index, nodes_avl.get(n)))

        # transition relation
        top_trans = poi_trans_count[cur_poi]["sorted"][:CONFIG.top_trans]
        # filtered_trans = [record for record in poi_trans_count[cur_poi]["sorted"] if record[0] >= 2]
        # top_trans = filtered_trans[:CONFIG.top_trans]
        for record in top_trans:
            if nodes_avl.get(record[1]) is None:
                nodes_avl.insert(record[1], len(graph_nodes))
                graph_nodes.append(record[1])
            trans_edges.add((cur_index, nodes_avl.get(record[1])))

        # same type relation
        same_type = poi_base.query_types(cur_poi, graph_nodes)
        for s in same_type:
            if nodes_avl.get(s) is None:
                nodes_avl.insert(s, len(graph_nodes))
                graph_nodes.append(s)
            s_edges.add((cur_index, nodes_avl.get(s)))
            s_edges.add((nodes_avl.get(s), cur_index))

    else:  # if the current poi is in the graph
        cur_index = nodes_avl.get(cur_poi)

        # transition relation
        top_trans = poi_trans_count[cur_poi]["sorted"][:CONFIG.top_trans]
        # filtered_trans = [record for record in poi_trans_count[cur_poi]["sorted"] if record[0] >= 2]
        # top_trans = filtered_trans[:CONFIG.top_trans]
        for record in top_trans:
            if nodes_avl.get(record[1]) is None:
                nodes_avl.insert(record[1], len(graph_nodes))
                graph_nodes.append(record[1])
            trans_edges.add((cur_index, nodes_avl.get(record[1])))

    return trans_edges, n_edges, s_edges, graph_nodes, nodes_avl


def merge_nodes(graph_nodes, add_nodes):
    for n in add_nodes:
        if n not in graph_nodes:
            graph_nodes.append(n)
    return graph_nodes


def processAllData(path, pp_dir="./data/weeplaces_pp"):
    # get data frame
    print("begin data processing.")
    if "weeplaces" in path:
        dtype_spec = {
            'userid': 'str',
            'placeid': 'str',
            'datetime': 'str',
            'lat': 'float64',
            'lon': 'float64',
            'city': 'str',
            'category': 'str'
        }
        df = pd.read_csv(path, dtype=dtype_spec)
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif "foursquare" in path:
        dtype_spec = {
            'userid': 'str',
            'placeid': 'str',
            'cateid': 'str',
            'category': 'str',
            'lat': 'float64',
            'lon': 'float64',
            'timezone': 'str',
            'datetime': 'str',
        }
        columns = ['userid', 'placeid', 'cateid', 'category', 'lat', 'lon', 'timezone', 'datetime']
        df = pd.read_csv(path, header=None, names=columns, dtype=dtype_spec, sep="\t", encoding="latin1")
        df['datetime'] = pd.to_datetime(df['datetime'], format='%a %b %d %H:%M:%S %z %Y')
    else:
        print("No such dataset")
        return
    print("opened successfully, processing...")
    df = df.dropna(subset=['placeid'])  # clear all rows that have no placeid
    df = df.dropna(subset=['category'])  # clear all rows that have no category
    df = df.sort_values(by='datetime')  # sort the data by time
    print("sorted by time.")

    # step one, get the poi and user information
    if not os.path.exists(pp_dir):
        os.makedirs(pp_dir)
    if load_step_1 and os.path.exists(os.path.join(pp_dir, "step_1_cache.pkl")):  # if this step is not processed
        print("load process step one.")
        with open(os.path.join(pp_dir, "step_1_cache.pkl"), 'rb') as file:
            poi_avl, user_avl, wastes_poi, wastes_user = pickle.load(file)
    else:
        print("begin process step one.")
        poi_info_list, user_info_list, poi_avl, user_avl, wastes_poi, wastes_user = processPOIandUsers(df)
        savePoiPoints(poi_info_list, os.path.join(pp_dir, "poi_points.csv"))
        with open(os.path.join(pp_dir, "step_1_cache.pkl"), 'wb') as file:
            print("save process step one cache.")
            pickle.dump((poi_avl, user_avl, wastes_poi, wastes_user), file)

    # step two, construct the database
    poi_base = POIBase(os.path.join(pp_dir, "poi_points.csv"), neighbor_radius=100)
    print("deleting useless poi...")
    df_filtered = df[~df['placeid'].isin(wastes_poi) & ~df['userid'].isin(wastes_user)]  # filter the wastes
    print("deleting useless user...")
    user_counts = df_filtered['userid'].value_counts()
    users_to_keep = user_counts[user_counts >= 5].index
    df_filtered = df_filtered[df_filtered['userid'].isin(users_to_keep)]

    print(f"{len(wastes_poi)} poi left")
    print(f"{len(wastes_user)} user left")
    print("{} rows left in dataframe.".format(len(df_filtered)))
    user_database = constructDatabase(df_filtered, poi_avl, user_avl, poi_base)

    saveDataset(user_database, os.path.join(pp_dir, "pped_v2.pkl"))


def saveDataset(dataset, save_path):
    print("deleting useless data structure to save space...")
    dataset_final = {}
    for user_id in tqdm(dataset):
        dataset_final[user_id] = {"graph_nodes": dataset[user_id]["graph_nodes"],
                                  "graph_split": dataset[user_id]["graph_split"],
                                  "traj": dataset[user_id]["traj"],
                                  "traj_split": dataset[user_id]["traj_split"],
                                  "time_stamps": dataset[user_id]["time_stamps"],
                                  "tvt_split": dataset[user_id]["tvt_split"]}

    with open(save_path, 'wb') as file:
        pickle.dump(dataset_final, file)
        print("saved dataset to {}.".format(save_path))
    return


def savePoiPoints(poi_list, save_path):
    target_file = open(save_path, "w")
    target_file.write("id,str_id,poi_name,category,interest,city,lon,lat\n")
    for record in tqdm(poi_list):
        target_file.write("{},{},{},{},{},{},{:.6f},{:.6f}\n".
                          format(record[0], record[1], record[2], record[3],
                                 record[4], record[5], record[6], record[7]))
    target_file.close()
    print("saved poi points to {}".format(save_path))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument("--dataset", type=str, default="nyc",
                        choices=["nyc", "tky", "cal", "flo"])
    args = parser.parse_args()
    CONFIG = init(args.dataset)
    if args.dataset == "cal":
        processAllData("./data/weeplaces/weeplace_california_checkins.csv", "./data/cal_pp")
    elif args.dataset == "flo":
        processAllData("./data/weeplaces/weeplace_florida_checkins.csv", "./data/flo_pp")
    elif args.dataset == "nyc":
        processAllData("./data/foursquare/dataset_tsmc2014/dataset_TSMC2014_NYC.txt", "./data/nyc_pp")
    elif args.dataset == "tky":
        processAllData("./data/foursquare/dataset_tsmc2014/dataset_TSMC2014_TKY.txt", "./data/tky_pp")
    else:
        print("No such dataset")
    print("done")
