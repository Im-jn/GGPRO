import argparse
import json
import os

from tqdm import tqdm

from config import CONFIG, init
from database import TrajDataset


def integrate_labels(labels_list):
    world_size = len(labels_list)
    # verify
    print("Verifying labels...")
    for i in range(world_size - 1):
        assert labels_list[i]["dataset_order"] == labels_list[i + 1]["dataset_order"]
    count = 0
    for i in range(world_size):
        count += len(labels_list[i]["labels"])
        assert labels_list[i]["rank"] == i
        assert labels_list[i]["total"] == world_size
    assert count == len(labels_list[0]["dataset_order"])
    print("Labels verified, total {} labels.".format(count))
    # integrate
    all_labels = {"dataset_order": labels_list[0]["dataset_order"], "labels": []}
    for i in range(world_size):
        all_labels["labels"].extend(labels_list[i]["labels"])
    return all_labels


def integrate_labels_with_dataset(pp_path, labels_list):
    dataset = TrajDataset(pp_path=pp_path,
                          action="train",
                          shuffle=CONFIG.shuffle)
    all_labels = {"data_index": dataset.data_index, "labels": [], "correct": []}
    label_count = 0
    for i in tqdm(range(len(dataset))):
        graph_nodes = dataset[i][2]
        assert len(graph_nodes) == len(labels_list[label_count]["labels"][0])
        assert labels_list[label_count]["idxs"][0] == i
        labels_list[label_count]["idxs"].pop(0)
        all_labels["labels"].append(labels_list[label_count]["labels"].pop(0))
        all_labels["correct"].append(labels_list[label_count]["correct"].pop(0))
        if len(labels_list[label_count]["labels"]) == 0:
            label_count += 1
            if label_count >= len(labels_list):
                print("Early stopping due to not enough labels!")
                return all_labels
    assert len(all_labels["labels"]) == len(dataset)
    assert len(all_labels["correct"]) == len(dataset)
    assert label_count == len(labels_list)
    assert len(labels_list[label_count-1]["labels"]) == 0
    return all_labels


def get_label_list(args):
    labels_list = []
    if args.specified_list is not None:
        print("Loading labels from specified list...")
        for item in args.specified_list:
            file_path = f"./data/GA_labels/{args.dataset}/{args.model_name.split('/')[-1]}/label_result_{item['rank']}_{item['world_size']}.json"
            with open(file_path, 'rb') as f:
                labels_list.append(json.load(f))
                assert labels_list[-1]["rank"] == item['rank']
                assert labels_list[-1]["total"] == item['world_size']
                print("Labels loaded from {}.".format(file_path))
    else:
        print("Loading labels from all list...")
        for i in range(args.world_size):
            file_path = f"./data/GA_labels/{args.dataset}_2/{args.model_name.split('/')[-1]}/label_result_{i}_{args.world_size}.json"
            with open(file_path, 'rb') as f:
                labels_list.append(json.load(f))
                assert labels_list[-1]["rank"] == i
                assert labels_list[-1]["total"] == args.world_size
                print("Labels loaded from {}.".format(file_path))
    return labels_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B",
                        choices=["Qwen/Qwen3-4B", "meta-llama/Meta-Llama-3-8B"])
    parser.add_argument("--dataset", type=str, default="nyc",
                        choices=["nyc", "tky", "cal", "flo"])
    parser.add_argument("--specified_list", type=list, default=None)
    args = parser.parse_args()
    CONFIG = init(args.dataset)
    pp_path = os.path.join("./data/{}_pp".format(args.dataset), "pped_v2.pkl")
    labels_list = get_label_list(args)
    all_labels = integrate_labels_with_dataset(pp_path, labels_list)
    final_file = f"./data/GA_labels/{args.dataset}_2/{args.model_name.split('/')[-1]}/all_labels.json"
    with open(final_file, 'w') as f:
        json.dump(all_labels, f)
        print("All labels saved to {}.".format(final_file))
