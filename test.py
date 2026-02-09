import argparse
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pickle
import re

import torch
from peft import PeftModel
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from config import CONFIG, init
from database import TrajDataset, POIBase
from small_models import NodesJudger, DummyJudger, ZeroJudger
from huggingface_hub import login

# same augmentation with RAG.py, but with special purpose
from prompt import poi_augment


def extract_answer(response):
    pattern = r"(\d+)"
    matches = [int(idx) for idx in re.findall(pattern, response[:10])]
    return matches


def do_judge(judger, state_graph, state_info, traj_info, imp_k=10):
    graph_poi_id_tensor = torch.tensor(state_info['id'].values, device="cuda:0")
    graph_cate_id_tensor = torch.tensor(state_info['category_id'].values, device="cuda:0")
    graph_lon_tensor = torch.tensor(state_info['lon'].values, dtype=torch.float32, device="cuda:0")
    graph_lat_tensor = torch.tensor(state_info['lat'].values, dtype=torch.float32, device="cuda:0")
    traj_poi_id_tensor = torch.tensor(traj_info['id'].values, device="cuda:0")
    traj_cate_id_tensor = torch.tensor(traj_info['category_id'].values, device="cuda:0")
    traj_lon_tensor = torch.tensor(traj_info['lon'].values, dtype=torch.float32, device="cuda:0")
    traj_lat_tensor = torch.tensor(traj_info['lat'].values, dtype=torch.float32, device="cuda:0")
    with torch.inference_mode():  # for testing
        action_embed, g_embed, c_score = judger(state_graph, (graph_poi_id_tensor, graph_cate_id_tensor, graph_lon_tensor, graph_lat_tensor),
                                                (traj_poi_id_tensor, traj_cate_id_tensor, traj_lon_tensor, traj_lat_tensor))
        importance_scores = F.cosine_similarity(g_embed, action_embed)

        pred = torch.sigmoid(importance_scores)
        # pred = importance_scores
        std = pred.std()
        mean = pred.mean()
        threshold = mean + temperature * std
        imp_k = imp_k if importance_scores.shape[-1] > imp_k else importance_scores.shape[-1]  # safe gaurd for important K
        top_nodes = torch.topk(importance_scores, k=imp_k).indices.tolist()
        important_nodes = torch.nonzero(importance_scores > threshold).squeeze(1).tolist()
        important_nodes = [i for i in top_nodes if i in important_nodes]  # choose the top 10 nodes that are important

    return importance_scores, important_nodes


def generate_prediction(prompt, tokenizer, llm_model, top_k=5):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    ).to(llm_model.device)
    # extract embeddings and make them trainable
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=None,
            num_beams=top_k,
            num_return_sequences=top_k,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            pad_token_id=llm_model.config.eos_token_id,
            early_stopping=True
        )
    # Decode the top-5 results of beam search
    decoded_texts = [
        tokenizer.decode(g[input_len:], skip_special_tokens=True) for g in outputs
    ]
    pred_visit = [extract_answer(t)[0] for t in decoded_texts]
    return pred_visit


def test(tokenizer, llm_model, judger, poi_base, dataset, args):
    top_1_acc, top_5_acc, top_10_acc, ndcg_5, ndcg_10, mrr, count = 0, 0, 0, 0, 0, 0, 0
    skipped = 0
    iters = len(dataset)
    progress_bar = tqdm(range(iters), desc=f"testing:", ncols=150)
    for i in progress_bar:
        if i == 200:
            print("mark!")
        elif (i+1) % 1000 == 0:
            print("mark!")
        user_id, graph, graph_nodes, traj, time_stamps, next_visit, next_time, last_traj_ptr = dataset[i]
        graph = graph.to("cuda:0")
        graph_info = poi_base.get_poi_info(graph_nodes)
        traj_info = poi_base.get_poi_info(traj)

        importance_scores, important_nodes = do_judge(judger, graph, graph_info, traj_info, imp_k=args.imp_k)
        # important_nodes = [i for i, v in enumerate(correct_label) if v == 1]

        prompt, poi_set, description_pieces = poi_augment(poi_base, user_id, graph, graph_nodes,
                                                          important_nodes,
                                                          traj, time_stamps, next_time,
                                                          last_traj_ptr,
                                                          with_answer_instruction=True,
                                                          with_hint=not args.mode == "no_hint")
        try:
            pred_visit = generate_prediction(prompt, tokenizer, llm_model, top_k=10)
        except Exception as e:
            print(f"Error during generation for user {user_id} at time {next_time}: {e}")
            skipped += 1
            continue
        if next_visit in pred_visit:
            rank = pred_visit.index(next_visit) + 1
            mrr += 1.0 / rank
            if rank == 1:
                top_1_acc += 1
            if rank <= 5:
                top_5_acc += 1
                ndcg_5 += 1.0 / np.log2(rank + 1)
            if rank <= 10:
                top_10_acc += 1
                ndcg_10 += 1.0 / np.log2(rank + 1)
        count += 1
        progress_bar.set_postfix({"acc 1": f"{top_1_acc/count:.4f}",
                                  "acc 5": f"{top_5_acc/count:.4f}",
                                  "acc 10": f"{top_10_acc/count:.4f}",
                                  "ndcg 5": f"{ndcg_5/count:.4f}",
                                  "ndcg 10": f"{ndcg_10/count:.4f}",
                                  "mrr": f"{mrr/count:.4f}",
                                  "skip": f"{skipped}"})
    return top_1_acc / count, top_5_acc / count, top_10_acc / count, ndcg_5 / count, ndcg_10 / count, mrr / count


class TrainDataset(Dataset):
    def __init__(self, poi_base, dataset, labels):
        self.poi_base = poi_base
        self.dataset = dataset
        self.labels = labels
        self.order = [i for i in range(len(dataset))]

    def __len__(self):
        return len(self.dataset)

    def set_loss_order(self, loss_records):
        self.order = np.argsort(loss_records)

    def __getitem__(self, idx):
        mapped_idx = self.order[idx]
        user_id, graph, graph_nodes, traj, time_stamps, next_visit, next_time = self.dataset[mapped_idx]
        graph = graph.to("cuda:0")
        graph_info = self.poi_base.get_poi_info(graph_nodes)
        traj_info = self.poi_base.get_poi_info(traj)
        label = self.labels["labels"][mapped_idx]
        correct = self.labels["correct"][mapped_idx]
        if correct:
            assert len(label) == len(graph_nodes)
        return user_id, graph, graph_info, traj_info, label, correct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B",
                        choices=["Qwen/Qwen3-4B", "meta-llama/Meta-Llama-3-8B"])
    parser.add_argument("--dataset", type=str, default="nyc",
                        choices=["nyc", "tky", "cal", "flo"])
    parser.add_argument("--mode", type=str, default="normal",
                        choices=["normal", "dummy", "random", "no_hint"])
    parser.add_argument("--base", action="store_true", default=False)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--imp_k", type=int, default=10)
    parser.add_argument("--ab_mode", type=str, default="")
    parser.add_argument("--ablation", action="store_true", default=False)
    parser.add_argument("--choice", type=str, default="valid", choices=["valid", "test", "train"])
    parser.add_argument("--new", action="store_true", default=False)
    args = parser.parse_args()
    CONFIG = init(args.dataset)

    poi_path = os.path.join("./data/{}_pp_2".format(args.dataset), "poi_points.csv")
    pp_path = os.path.join("./data/{}_pp_2".format(args.dataset), "pped_v2.pkl")
    poi_base = POIBase(poi_path)
    valid_dataset = TrajDataset(pp_path=pp_path,
                                action=args.choice,
                                shuffle=CONFIG.shuffle)

    if args.mode == "dummy":  # use a dummy judger that always outputs zeros
        print("using DummyJudger")
        judger = DummyJudger(CONFIG.gnn_in_feat, CONFIG.gnn_hid_feat, 3, len(poi_base.pois_df), len(poi_base.cate_list), 
                             poi_base.bbox, dropout=0.25).to("cuda:0")
    elif args.mode == "random":  # don't load pre-trained weights
        print("using RandomJudger")
        judger = NodesJudger(CONFIG.gnn_in_feat, CONFIG.gnn_hid_feat, 3, len(poi_base.pois_df), len(poi_base.cate_list), 
                             poi_base.bbox, dropout=0.25).to("cuda:0")
    elif args.mode == "no_hint":  # use a zero judger that always outputs zeros
        print("using ZeroJudger")
        judger = ZeroJudger(CONFIG.gnn_in_feat, CONFIG.gnn_hid_feat, 3,
                             len(poi_base.pois_df), len(poi_base.cate_list), dropout=0.25).to("cuda:0")
    else:
        judger = NodesJudger(CONFIG.gnn_in_feat, CONFIG.gnn_hid_feat, 3, len(poi_base.pois_df), len(poi_base.cate_list), 
                             poi_base.bbox, dropout=0.25).to("cuda:0")
        if args.epoch != -1:
            file_name = f"judger_epoch{args.epoch}.pth"
        else:
            # file_name = "judger_best.pth"
            file_name = f"judger_best_new.pth" if args.new else "judger_best.pth"
        if args.ablation:
            state_dict = torch.load(f"./checkpoints/ablation/{args.dataset}/{args.model_name.split('/')[-1]}/{file_name}")
        else:
            state_dict = torch.load(f"./checkpoints/{args.dataset}/{args.model_name.split('/')[-1]}/{file_name}")
        judger.load_state_dict(state_dict, strict=False)
        judger.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16,
                                              local_files_only=False)
    llm_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16,
                                                     local_files_only=False).to("cuda:0")
    print(f"Loaded base LLM model {args.model_name}.")
    if not args.base:
        peft_model_dir = f"./data/{args.dataset}_out/{args.model_name.split('/')[-1]}/model"
        llm_model = PeftModel.from_pretrained(llm_model, peft_model_dir).to("cuda:0")
        print(f"Loaded PEFT model from {peft_model_dir}")
    llm_model.eval()
    # Final evaluation on validation set
    print("Final evaluation on validation set:")
    acc_1, acc_5, acc_10, ndcg_5, ndcg_10, mrr = test(tokenizer, llm_model, judger, poi_base, valid_dataset, args)
    print(f"Top 1 Acc: {acc_1:.4f}, Top 5 Acc: {acc_5:.4f}, Top 10 Acc: {acc_10:.4f}, "
          f"NDCG@5: {ndcg_5:.4f}, NDCG@10: {ndcg_10:.4f}, MRR: {mrr:.4f}")

