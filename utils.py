import dgl
import numpy as np
import math
import pickle
import torch
import io
import networkx as nx
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class geo_math:
    def __init__(self):
        self.radius = 6378137.0  # Semi-major axis in meters.
        e = 8.1819190842622e-2  # First eccentricity.

    def llh_to_ecef(self, lon, lat, alt=0):
        # WGS 84 ellipsoid parameters.

        # Convert longitude and latitude from degrees to radians.
        lon = np.radians(lon)
        lat = np.radians(lat)

        # Compute ECEF coordinates.
        N = self.radius
        x = (N + alt) * np.cos(lat) * np.cos(lon)
        y = (N + alt) * np.cos(lat) * np.sin(lon)
        z = (N + alt) * np.sin(lat)
        return x, y, z

    def arc_length_to_chord_length(self, arc_length):
        # Compute the central angle in radians.
        theta = arc_length / self.radius
        # Compute the chord length.
        chord_length = 2 * self.radius * math.sin(theta / 2)

        return chord_length

    def haversine_distance(self, lon1, lat1, lon2, lat2):
        # Convert decimal degrees to radians.
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

        # Haversine formula.
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Earth radius in meters.
        distance = self.radius * c

        return distance


def maximum_spanning_tree(graph, node_scores):  # get the maximum spanning tree of a simplified graph
    hg = dgl.to_homogeneous(graph)
    ug = dgl.to_bidirected(hg)
    ug_simple = dgl.to_simple(ug, copy_ndata=True, aggregator='mean')  # simplify graph
    edge_index = ug_simple.edges()  # get edge index
    src, dst = edge_index   # get source and destination nodes of edges
    w = (node_scores[src] + node_scores[dst]) / 2.0

    # For duplicate undirected edges, keep only the edge with the largest weight.
    best = {}
    for u, v, weight in zip(src.tolist(), dst.tolist(), w.tolist()):
        a, b = (u, v) if u <= v else (v, u)
        if (a, b) not in best or weight > best[(a, b)][0]:
            best[(a, b)] = (weight, u, v)

    G = nx.Graph()  # create a NetworkX graph
    for (a, b), (weight, u, v) in best.items():
        G.add_edge(u, v, weight=float(weight))  # add edge with weight
    mst = nx.maximum_spanning_tree(G, weight="weight")  # compute maximum spanning tree
    return list(mst.edges())


def top_k_edges(graph, node_scores, top_k=20):  # get the maximum spanning tree of a simplified graph
    hg = dgl.to_homogeneous(graph)
    ug = dgl.to_bidirected(hg)
    ug_simple = dgl.to_simple(ug, copy_ndata=True, aggregator='mean')  # simplify graph
    u, v = ug_simple.edges()  # (E,), (E,)
    w = (node_scores[u] + node_scores[v]) / 2.0  # (E,)
    E = ug_simple.num_edges()
    k_eff = min(top_k, E)
    topw, idx = torch.topk(w, k=k_eff, largest=True, sorted=True)  # (k_eff,)
    topk_eids = idx.tolist()
    topk_pairs = [(int(u[i]), int(v[i])) for i in topk_eids]
    return topk_pairs

