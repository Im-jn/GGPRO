import argparse
import os
import json
from datetime import datetime
import torch


def GetArgs():
    args = DotDict({
        "action": 'train',
        "device": 0,
        "LLM_device": 0,
        "valid_device": 1,
        "dataset": 'nyc',
        "experiment": 0,
        "load": False,
        "resume": -1,
        "accelerate": False,
    })
    
    return args


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


def load_dataset_config(dataset_name=None):
    """
    Load dataset-specific configuration file
    
    Args:
        dataset_name: Dataset name (nyc, tky, wee), if None, return default configuration
    
    Returns:
        DotDict: Configuration dictionary
    """
    if dataset_name is None:
        return None
    
    config_path = os.path.join("./data/config", f"{dataset_name}_config.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            # Convert tuple to list (JSON does not support tuple)
            if "tvt_rate" in config_dict and isinstance(config_dict["tvt_rate"], list):
                config_dict["tvt_rate"] = tuple(config_dict["tvt_rate"])
            return DotDict(config_dict)
        except Exception as e:
            print(f"[WARNING] Failed to load config from {config_path}: {e}")
            print(f"[WARNING] Using default config instead.")
            return None
    else:
        print(f"[WARNING] Config file not found: {config_path}")
        print(f"[WARNING] Using default config instead.")
        return None


def init(dataset_name=None):  # init config with optional dataset-specific config
    # Load default configuration first (keep only actually used configuration items)
    CONFIG = DotDict({
        # data prepare
        "top_trans": 10,  # top 10 all user transition (add to graph)
        "min_count": 10,  # the acceptable user and POI numbers (should be more than this number)
        "shuffle": True,  # shuffle dataset, the dataset will be shuffled according to "random_seed"
        "tvt_rate": (0.8, 0.1, 0.1),  # train/valid/test split ratio

        # dataset
        "dataset_name": "Not Yet Initiated",

        # paths & dirs
        "model_path": "Not Yet Initiated",
        "memory_path": "Not Yet Initiated",
        "data_pp_dir": "Not Yet Initiated",
        "data_out_dir": "Not Yet Initiated",
        "peft_model_dir": "Not Yet Initiated",

        # agent (GNN model settings)
        "gnn_in_feat": 128,
        "gnn_hid_feat": 256,
        "accumulation_steps": 16,  # gradient accumulation steps

        # prompt
        "random_seed": 2025,
        "important_nodes": 10,  # select the important trajectory nodes to describe
        "recent_visits": 10,  # number of recent visits to include in prompt
        "history_visits": 32,  # number of history visits to include in prompt
        "curtail_graph": 64,  # maximum number of nodes in graph
    })
    
    # If a dataset name is specified, try to load the dataset-specific configuration
    if dataset_name:
        dataset_config = load_dataset_config(dataset_name)
        if dataset_config:
            # Override default configuration with dataset-specific configuration
            CONFIG.update(dataset_config)
            print(f"[CONFIG] Loaded dataset-specific config for: {dataset_name}")
    print_config(CONFIG)
    return CONFIG


def update_config_with_dataset(dataset_name):
    """
    Update global CONFIG to the configuration of the specified dataset
    
    Args:
        dataset_name: Dataset name (nyc, tky, wee)
    
    Returns:
        bool: Whether the configuration is successfully loaded
    """
    global CONFIG
    dataset_config = load_dataset_config(dataset_name)
    if dataset_config:
        CONFIG.update(dataset_config)
        print(f"[CONFIG] Updated global CONFIG with dataset-specific config for: {dataset_name}")
        return True
    else:
        print(f"[CONFIG] Failed to load config for dataset: {dataset_name}")
        return False


def print_config(CONFIG):
    # print config (only print actual used configuration items)
    print("TIME: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("[CONFIG] Dataset: {}".format(CONFIG.dataset_name))
    print("[CONFIG] Data PP Dir: {}".format(CONFIG.data_pp_dir))
    print("[CONFIG] Top Trans: {}".format(CONFIG.top_trans))
    print("[CONFIG] Min Count: {}".format(CONFIG.min_count))
    print("[CONFIG] Data Out Dir: {}".format(CONFIG.data_out_dir))
    print("[CONFIG] PEFT Model Dir: {}".format(CONFIG.peft_model_dir))
    print("[CONFIG] Shuffle: {}".format(CONFIG.shuffle))
    print("[CONFIG] Random Seed: {}".format(CONFIG.random_seed))
    print("[CONFIG] Important Nodes: {}".format(CONFIG.important_nodes))
    print("[CONFIG] History Visits: {}".format(CONFIG.history_visits))
    print("[CONFIG] Curtail Graph: {}".format(CONFIG.curtail_graph))
    print("[CONFIG] GNN Input Features: {}".format(CONFIG.gnn_in_feat))
    print("[CONFIG] GNN Hidden Features: {}".format(CONFIG.gnn_hid_feat))
    print("[CONFIG] Accumulation Steps: {}".format(CONFIG.accumulation_steps))



# configuration for data preparing
# Default initialization, can be loaded at runtime via load_dataset_config or by calling init(dataset_name) again to load specific dataset configuration
CONFIG = init()
