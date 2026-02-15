import argparse
import json
import os

import dgl
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from small_models import NodesJudger

from prompt import poi_augment


def extract_answer(response):
    pattern = r"(\d+)"
    matches = [int(idx) for idx in re.findall(pattern, response[:10])]
    return matches


def do_judge(judger, state_graph, state_info, traj_info):
    graph_poi_id_tensor = torch.tensor(state_info['id'].values, device="cuda:0")
    graph_cate_id_tensor = torch.tensor(state_info['category_id'].values, device="cuda:0")
    graph_lon_tensor = torch.tensor(state_info['lon'].values, dtype=torch.float32, device="cuda:0")
    graph_lat_tensor = torch.tensor(state_info['lat'].values, dtype=torch.float32, device="cuda:0")
    traj_poi_id_tensor = torch.tensor(traj_info['id'].values, device="cuda:0")
    traj_cate_id_tensor = torch.tensor(traj_info['category_id'].values, device="cuda:0")
    traj_lon_tensor = torch.tensor(traj_info['lon'].values, dtype=torch.float32, device="cuda:0")
    traj_lat_tensor = torch.tensor(traj_info['lat'].values, dtype=torch.float32, device="cuda:0")
    action_embed, g_embed, c_score = judger(state_graph, (graph_poi_id_tensor, graph_cate_id_tensor, graph_lon_tensor, graph_lat_tensor),
                                            (traj_poi_id_tensor, traj_cate_id_tensor, traj_lon_tensor, traj_lat_tensor))
    importance_scores = F.cosine_similarity(g_embed, action_embed)

    pred = torch.sigmoid(importance_scores)
    # pred = importance_scores
    std = pred.std()
    mean = pred.mean()
    threshold = mean + 1.5 * std
    important_nodes = torch.nonzero(importance_scores > threshold).squeeze(1).tolist()
    return importance_scores, important_nodes


def recall_regularizer(logits, labels, recall_target=0.2, lam=0.5):
    probs = torch.sigmoid(logits)
    recall_soft = (probs * labels).sum() / (labels.sum() + 1e-6)
    return lam * F.relu(recall_target - recall_soft)


def my_loss(
    scores: torch.Tensor,
    pos_mask: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-6,
):
    """
    Multiple-Instance Learning (MIL) loss for attribution-based evidence selection.

    Args:
        scores: Tensor [num_nodes], raw scores/logits from the judger
        pos_mask: Bool or 0/1 Tensor [num_nodes], attribution-positive nodes
        temperature: softmax temperature for smoothing (>=1.0 is safer)
        eps: numerical stability

    Returns:
        loss: scalar Tensor
        stats: dict for logging
    """

    # safety: no positive or no negative → no supervision
    if pos_mask.sum() == 0 or pos_mask.sum() == scores.numel():
        zero = scores.sum() * 0.0
        return zero

    # positive / negative bags
    pos_scores = scores[pos_mask] / temperature
    neg_scores = scores[~pos_mask] / temperature

    # bag-level scores (log-sum-exp pooling)
    bag_pos = torch.logsumexp(pos_scores, dim=0)
    bag_neg = torch.logsumexp(neg_scores, dim=0)

    # logistic ranking loss
    loss = -torch.log(torch.sigmoid(bag_pos - bag_neg) + eps)

    return loss

def training(judger, optimizer, dataset, labels, save_loss=False):
    loss_fn = nn.CrossEntropyLoss()
    steps = 0
    loss_records = []
    progress_bar = tqdm(range(8863, len(dataset)), desc=f"Training:", ncols=150)
    for i in progress_bar:
        user_id, graph, graph_info, traj_info, label, correct = dataset[i]
        if correct and sum(label) > 0 and len(traj_info) > 0:  # if the data need to be trained
            try:
                importance_scores, _ = do_judge(judger, graph, graph_info, traj_info)
                # label = torch.tensor(label, dtype=torch.float32, device=importance_scores.device)
                # judger_loss = loss_fn(importance_scores, label)
                label = torch.tensor(label, dtype=torch.int64, device=importance_scores.device)
                judger_loss = my_loss(importance_scores, label)
                accum_loss = judger_loss / CONFIG.accumulation_steps
                accum_loss.backward()
            except Exception as e:
                print(f"Error during training for iter {i}: {e}")
                continue
            steps += 1

            if steps >= CONFIG.accumulation_steps or (i + 1) == len(dataset):
                optimizer.step()
                optimizer.zero_grad()
                steps = 0

            loss_records.append(judger_loss.item())
            progress_bar.set_postfix({"loss": f"{judger_loss.item():.4f}",
                                      "avg_loss": f"{(sum(loss_records[-1000:]) / len(loss_records[-1000:])):.4f}",
                                      "correct": f"{sum(labels['correct']) / len(labels['correct']):.4f}"})
        else:
            continue
    if save_loss:
        with open("./data/loss_records.pkl", "wb") as f:
            pickle.dump(loss_records, f)
    return judger


def valid(llm_model, judger, poi_base, dataset):
    top_1_acc, top_5_acc, error_count, count = 0, 0, 0, 0
    iters = 100
    progress_bar = tqdm(range(iters), desc=f"Validing:", ncols=150)
    for i in progress_bar:
        user_id, graph, graph_nodes, traj, time_stamps, next_visit, next_time, last_traj_ptr = dataset[i]
        graph = graph.to("cuda:0")
        graph_info = poi_base.get_poi_info(graph_nodes)
        traj_info = poi_base.get_poi_info(traj)

        _, important_nodes = do_judge(judger, graph, graph_info, traj_info)

        prompt, poi_set, description_pieces = poi_augment(poi_base, user_id, graph, graph_nodes,
                                                              important_nodes,
                                                              traj, time_stamps, next_time,
                                                              last_traj_ptr,
                                                              with_answer_instruction=True)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(llm_model.device)
        # extract embeddings and make them trainable
        input_len = inputs["input_ids"].shape[1]
        try:
            with torch.inference_mode():
                outputs = llm_model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=None,
                    num_beams=5,
                    num_return_sequences=5,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    pad_token_id=llm_model.config.eos_token_id,
                    early_stopping=True
                )
            # 将 beam search 的 top-5 结果 decode 出来
            decoded_texts = [
                tokenizer.decode(g[input_len:], skip_special_tokens=True) for g in outputs
            ]
            pred_visit = [extract_answer(t)[0] for t in decoded_texts]
            top_1_acc += (next_visit == pred_visit[0])
            top_5_acc += (next_visit in pred_visit)
            count += 1
            progress_bar.set_postfix({"top 1 acc": f"{top_1_acc/count:.4f}",
                                    "top 5 acc": f"{top_5_acc/count:.4f}",
                                    "error": f"{error_count}"})
        except Exception as e:
            print(f"Error during generation for user {user_id} at time {next_time}: {e}")
            error_count += 1
            progress_bar.set_postfix({"top 1 acc": f"{top_1_acc/count:.4f}",
                                      "top 5 acc": f"{top_5_acc/count:.4f}", 
                                      "error": f"{error_count}"})
    return top_1_acc / iters


class TrainDataset(Dataset):
    def __init__(self, poi_base, dataset, labels):
        self.poi_base = poi_base
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        user_id, graph, graph_nodes, traj, time_stamps, next_visit, next_time, last_traj_ptr = self.dataset[idx]
        graph_info = self.poi_base.get_poi_info(graph_nodes)
        traj_info = self.poi_base.get_poi_info(traj)
        label = self.labels["labels"][idx]
        correct = self.labels["correct"][idx]
        if correct:  # safty check!
            if len(label) != len(graph_nodes):
                print("error!")
            assert len(label) == len(graph_nodes)
        if next_visit in graph_nodes:
            idx = graph_nodes.tolist().index(next_visit)
            label[idx] = 1
        graph = graph.to("cuda:0")
        return user_id, graph, graph_info, traj_info, label, correct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B",
                        choices=["Qwen/Qwen3-4B", "meta-llama/Meta-Llama-3-8B"])
    parser.add_argument("--dataset", type=str, default="nyc",
                        choices=["nyc", "tky", "cal", "flo"])
    parser.add_argument("--resume_epoch", type=int, default=0)
    parser.add_argument("--total_epochs", type=int, default=3)
    parser.add_argument("--new", action="store_true", default=False)
    args = parser.parse_args()
    CONFIG = init(args.dataset)

    poi_path = os.path.join("./data/{}_pp".format(args.dataset), "poi_points.csv")
    pp_path = os.path.join("./data/{}_pp".format(args.dataset), "pped_v2.pkl")
    label_file = f"./data/GA_labels/{args.dataset}/{args.model_name.split('/')[-1]}/all_labels.json"
    with open(label_file, 'rb') as f:
        labels = json.load(f)
        print(f"Loaded labels from {label_file}")
    poi_base = POIBase(poi_path)
    train_dataset = TrajDataset(pp_path=pp_path,
                                action="train",
                                shuffle=CONFIG.shuffle,)
    valid_dataset = TrajDataset(pp_path=pp_path,
                                action="valid",
                                shuffle=CONFIG.shuffle)

    train_dataset = TrainDataset(poi_base, train_dataset, labels)

    judger = NodesJudger(CONFIG.gnn_in_feat, CONFIG.gnn_hid_feat, 5, len(poi_base.pois_df), len(poi_base.cate_list), 
                         poi_base.bbox, dropout=0.15).to("cuda:0")
    ## Training setup
    lr_start = 5e-4
    gamma = 0.5
    lr = lr_start * (gamma ** args.resume_epoch)
    optimizer = torch.optim.AdamW(judger.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16,
                                              local_files_only=False)
    llm_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16,
                                                     local_files_only=False)
    peft_model_dir = f"./data/{args.dataset}_out/{args.model_name.split('/')[-1]}/model"
    print(f"Loaded PEFT model from {peft_model_dir}")
    llm_model = PeftModel.from_pretrained(llm_model, peft_model_dir).to("cuda:0")
    llm_model.eval()

    best_result = 0.0
    checkpoint_dir = f"./checkpoints/{args.dataset}/{args.model_name.split('/')[-1]}"
    if not os.path.exists(checkpoint_dir):  # check whether the checkpoint directory exists
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory. {checkpoint_dir}")

    if args.resume_epoch > 0:
        judger.load_state_dict(torch.load(f"{checkpoint_dir}/judger_epoch{args.resume_epoch}.pth"))
        print(f"Resumed from epoch {args.resume_epoch}.")

    for epoch in range(args.resume_epoch, args.total_epochs):
        print(f"############## Epoch {epoch + 1}/{args.total_epochs} ##############")
        judger.train()
        judger = training(judger, optimizer, train_dataset, labels)
        scheduler.step()  # lr decay
        if args.new:
            torch.save(judger.state_dict(), f"{checkpoint_dir}/judger_epoch{epoch + 1}_new.pth")
        else:
            torch.save(judger.state_dict(), f"{checkpoint_dir}/judger_epoch{epoch + 1}.pth")
        print("Model checkpoint saved.")
        judger.eval()
        result = valid(llm_model, judger, poi_base, valid_dataset)
        if result > best_result:
            best_result = result
            torch.save(judger.state_dict(), f"{checkpoint_dir}/judger_best.pth")
            print(f"New best model saved to '{checkpoint_dir}' with top-1 acc: {best_result:.4f}")

    # # Final evaluation on validation set
    # print("Final evaluation on validation set:")
    # judger.load_state_dict(torch.load(f"{checkpoint_dir}/judger_best_new.pth"))
    # judger.eval()
    # valid(llm_model, judger, poi_base, valid_dataset)

