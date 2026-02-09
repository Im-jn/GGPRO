################################
# Use iteration to find label
# Use beamsearch to retrieve output
# Save result label in json file
###############################

import numpy as np
import pydevd_pycharm
from tqdm import tqdm

# print("E04.py")
# pydevd_pycharm.settrace('0.0.0.0', port=12345, stdoutToServer=True, stderrToServer=True)

import argparse
import json
import os
import random
import re
import shutil
import time
from collections import OrderedDict
import copy
import deepspeed

import torch
import torch.distributed as dist
from huggingface_hub import login
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from database import TrajDataset, POIBase
from prompt import poi_augment
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
from peft import PeftModel, PeftConfig
from safetensors import safe_open
from config import CONFIG, print_config, init

# ### Suppress FutureWarnings ###
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
# ### Note: This is to avoid FutureWarnings that may clutter the output ###


def clear_dir(dir_path):
    if os.path.isdir(dir_path):
        files = os.listdir(dir_path)
    else:
        print("Log directory is clear.")
        return
    if len(files) > 0:
        # command = input("There are files in the log directory '{}'. Clear the files?(yes/no): \n".format(dir_path))
        print("There are files in the log directory '{}'. Clear the files?(yes/no): ".format(dir_path))
        command = "yes"
        print(command + "(auto)")
        if command == "yes":
            for file_name in files:
                file_path = os.path.join(dir_path, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)  # Delete file or symbolic link
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            print("Log directory is now cleared.")
        else:
            print("Sure, never mind.")
    else:
        print("Log directory is clear.")


# screen out the token places of the concerned sentence
def screen_token_place(source, target, offset_mapping):
    start_idx = source.find(target)
    end_idx = start_idx + len(target)
    token_indices = [
        i for i, (start, end) in enumerate(offset_mapping)
        if start < end_idx and end > start_idx
    ]
    return token_indices


# tokenize all the prompt and label
def process_tokenize(tokenizer, prompt, description_pieces, label, device="cpu"):
    screen_position = {"graph": [],
                    #    "history": [],
                       "prompt_mask": [],
                       "label_mask": []}
    sentence = prompt + label
    inputs = tokenizer(
        sentence,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        # max_length=1024,
    ).to(device)
    offset_mapping = inputs["offset_mapping"][0]

    # find graph prompt place
    for graph_prompt in description_pieces["graph"]:
        screen_position["graph"].append(screen_token_place(sentence, graph_prompt, offset_mapping))
    # screen_position["history"] = screen_token_place(sentence, description_pieces["history"], offset_mapping)
    screen_position["prompt_mask"] = screen_token_place(sentence, prompt, offset_mapping)
    screen_position["label_mask"] = list(range(screen_position["prompt_mask"][-1], inputs["input_ids"].shape[-1]-1))
    # label mask is 1 token shifted compared to input_ids, because the output is always the next token of input

    return inputs, screen_position


def minus(index_list):
    ret_list = []
    for i in index_list:
        if i != 0:
            ret_list.append(i - 1)
    return ret_list


def plus(index_list):
    ret_list = []
    for i in index_list:
        if i != 0:
            ret_list.append(i + 1)
    return ret_list


# process the token embeddings and screen out the concerned embeddings
def process_embeddings(model, input_ids, screen_position, get_screen=True, adapt_emb=None):
    em_layer = model.get_input_embeddings()
    label = torch.ones_like(input_ids) * -100
    label[:, screen_position["label_mask"]] = input_ids.clone()[:, plus(screen_position["label_mask"])]

    screen_tensor = {"graph": [],
                     "history": None}
    with torch.no_grad():  # Freeze model, no gradient needed
        input_embedding = em_layer(input_ids)

    if get_screen:
        # make screen tensors with grad
        for g_screen in screen_position["graph"]:
            st = input_embedding[:, g_screen, :].detach().clone().requires_grad_()
            # st = input_embedding[:, g_screen, :].detach().clone()
            screen_tensor["graph"].append(st)
            input_embedding[:, g_screen, :] = st
        # screen_tensor["history"] = input_embedding[:, screen_position["history"], :].detach().clone().requires_grad_()
        # input_embedding[:, screen_position["history"], :] = screen_tensor["history"]

    return input_embedding, label, screen_tensor


def freeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model


def pool_weight(weight):
    with torch.no_grad():
        abs_weight = torch.abs(weight)
        return abs_weight.mean(dim=-1)


def max_weight(weight):
    with torch.no_grad():
        abs_weight = torch.abs(weight)
        return abs_weight.max(dim=-1).values


def tensor_weight(grad, tensor, p=2):
    with torch.no_grad():
        weight = grad * tensor
        return weight.norm(p=p, dim=-1)
        # return weight.sum(dim=-1)

def contrast_weight(src, src_grad, tgt, tgt_grad):
    with torch.no_grad():
        # weight = src * src_grad - tgt * tgt_grad
        # return weight.norm(p=p, dim=-1)
        # # return weight.sum(dim=-1)
        # v1 = src * src_grad
        # v2 = tgt * tgt_grad
        # # dim=-1 means calculation on the last dimension, keepdim is optional
        remain = remove_common_part(src_grad, tgt_grad)
        ret = (remain * src).norm(p=2, dim=-1)
        return ret


def remove_common_part(A, B, eps=1e-12):
    # A, B: [*, D] tensors of same shape
    dot = (B * A).sum(dim=-1, keepdim=True)   # [*, 1]
    norm_sq = A.norm(p=2, dim=-1, keepdim=True) + eps
    proj = dot / norm_sq * A
    residual = B - proj
    return residual


def sum_weight(grad, tensor):
    with torch.no_grad():
        weight = grad * tensor
        # return weight.norm(p=p, dim=-1)
        return weight.sum(dim=-1)



def sentence_weight(token_weight):
    with torch.no_grad():
        # return token_weight.mean(dim=-1)
        top_n = 20 if token_weight.shape[-1] >= 20 else token_weight.shape[-1]
        return float(token_weight.topk(k=top_n).values.mean(dim=-1))


def cal_all_weight(screen_tensor=None):
    weights = []
    for p in screen_tensor["graph"]:
        t_weight = tensor_weight(p.grad, p)
        weights.append(sentence_weight(t_weight))
    # benchmark = tensor_weight(screen_tensor["history"].grad, screen_tensor["history"])
    return weights


def cal_contrast_weight(src_screen, tgt_screen):
    weights = []
    for src, tgt_dict in zip(src_screen, tgt_screen):
        src_grad = src.grad
        tgt = tgt_dict["val"]
        tgt_grad = tgt_dict["grad"]
        c_weight = contrast_weight(src, src_grad, tgt, tgt_grad)
        weights.append(sentence_weight(c_weight))
    return weights


def mark_outliers(data, k=2.0):
    arr = np.array(data)
    mean = arr.mean()
    std = arr.std()

    threshold_1 = mean + k * std
    threshold_2 = mean - k * std
    result = [1 if (x > threshold_1 or x < threshold_2) else 0 for x in arr]
    return result

def mark_inliners(data, k=1):
    arr = np.array(data)
    mean = arr.mean()
    std = arr.std()

    threshold_1 = mean + k * std
    threshold_2 = mean - k * std
    result = [1 if (threshold_2 < x < threshold_1) else 0 for x in arr]
    return result


def normalize(tensor):
    min_value = tensor.min()
    max_value = tensor.max()
    assert min_value != max_value
    return (tensor - min_value) / (max_value - min_value)


def readable_output(outputs, tokenizer, true_mask):
    with torch.no_grad():
        logits = outputs.logits
        token_ids = torch.argmax(logits, dim=-1)
        predicted_text = tokenizer.decode(token_ids[0][true_mask], skip_special_tokens=True)
    return predicted_text


def extract_answer(response):
    pattern = r"(\d+)"
    matches = [int(idx) for idx in re.findall(pattern, response[:10])]
    return matches


def screen_zero_grad(screen_tensor):
    for i in range(len(screen_tensor["graph"])):
        screen_tensor["graph"][i].grad = None
    # screen_tensor["history"].grad = None


def log_softmax(logits):
    logits = logits - logits.max(dim=-1, keepdim=True)[0]  # avoid overflow
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # cal log_softmax
    return log_probs


def clone_grad_tensor(screen_tensor):
    backup = []
    for t in screen_tensor["graph"]:
        content = {"val": t.detach().clone(), "grad": t.grad.detach().clone()}
        backup.append(content)
    return backup


def zero_grad(model_engine):
    # zero grad for model engine
    for p in model_engine.parameters():
        if p.grad is not None:
            p.grad = None
    return model_engine


def improvement_statistic(improvement_list):
    count = 0
    avg_improvement = 0
    for final_rank, avg_rank in improvement_list:
        avg_improvement += (avg_rank - final_rank)
        if final_rank <= avg_rank:
            count += 1
    print("{} out of {} improved".format(count, len(improvement_list)))
    print("average improvement: {:.4f}".format(avg_improvement / len(improvement_list)))
    return


def save_checkpoint(checkpoint_path, iter_count, saving_data,
                    test_1_acc, test_5_acc, test_count, skip_count):
    """Save checkpoint for current loop"""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_data = {
        "iter_count": iter_count,
        "saving_data": saving_data,
        "test_1_acc": test_1_acc,
        "test_5_acc": test_5_acc,
        "test_count": test_count,
        "skip_count": skip_count,
    }

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)
        print("successfully saved checkpoint.")


def load_checkpoint(checkpoint_path):
    """Load checkpoint, return None if not exists"""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        print(f"Checkpoint loaded from iter_count={checkpoint_data['iter_count']}")
        return checkpoint_data
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None

def extra_hint(graph, graph_nodes, next_visit):
    if next_visit not in graph_nodes:
        return None
    # 1. Find the node index corresponding to next_visit
    center_idx = graph_nodes.index(next_visit)

    extra_hint_nodes = set()
    extra_hint_nodes.add(center_idx)

    # 2. Traverse all edge types related to poi
    for etype in graph.canonical_etypes:
        # Out edge: next_visit -> others
        succ = graph.successors(center_idx, etype=etype)
        extra_hint_nodes.update(succ.tolist())

        # In edge: others -> next_visit
        pred = graph.predecessors(center_idx, etype=etype)
        extra_hint_nodes.update(pred.tolist())

    return list(extra_hint_nodes)

def judge_have_extra_hint(extra_hint_nodes, imp_nodes):
    if extra_hint_nodes is None:
        return True
    for node in imp_nodes:
        if node in extra_hint_nodes:
            return True
    return False


def testify(model, tokenizer, inputs, args):
    try:
        # extract embeddings and make them trainable
        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                top_k=None,
                temperature=1.0,
                top_p=1.0,
                num_beams=args.beam_size,
                num_return_sequences=args.beam_size,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                pad_token_id=model.config.eos_token_id,
                early_stopping=True
            )
        # Decode the top-5 results of beam search
        decoded_texts = [
            tokenizer.decode(g[input_len:], skip_special_tokens=True) for g in outputs
        ]
        pred_visit = [extract_answer(t)[0] for t in decoded_texts]
        return pred_visit
    except RuntimeError as e:
        print("RuntimeError:", e)
        return []


def llm_process(args, pp_path, poi_path):
    # this should be a replacement for the method in RL_new.py
    # idea for this process
    # 1. use RAG for prompt generation;
    # 2. find the position in tokens of the sentences for attribution;
    # 3. extract embeddings and make them trainable;
    # 4. calculate loss;
    # 5. backward LLM and calculate contribution tensor;
    # 6. train GNN with contribution tensor;

    multiprocess = (args.p_rank, args.p_total)
    model_name, dataset_name = args.model_name, args.dataset
    poi_base = POIBase(poi_path)
    dataset = TrajDataset(pp_path=pp_path,
                          action=args.mode,
                          shuffle=CONFIG.shuffle)
    print("{} records in {} dataset".format(len(dataset), args.mode))
    content = {'model': 'llm', 'tag': "begin", 'total_steps': '{}'.format(len(dataset))}
    print("[PROGRESS] "+json.dumps(content), flush=True)
    
    # Set checkpoint path
    checkpoint_dir = f"./data/GA_labels/{dataset_name}_2/{model_name.split('/')[-1]}/checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{args.mode}_{args.p_rank}_{args.p_total}.json")
    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16,
                                              local_files_only=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                 local_files_only=False)
    model = PeftModel.from_pretrained(model, f"./data/{dataset_name}_out/{model_name.split('/')[-1]}/model")
    model.gradient_checkpointing_enable()
    # model = freeze_model(model)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        # model_parameters=model.parameters(),
        optimizer=None,
        model_parameters=None,
        lr_scheduler=None,
    )
    model_engine.eval()
    print("model_engine.optimizer:", model_engine.optimizer)
    acc, acc_1, acc_2, last_acc, a_loss = 0, 0, 0, 0, 0
    test_1_acc, test_5_acc, test_count, skip_count = 0, 0, 0, 0
    skip_final_test = False
    saving_data = {"dataset_order": dataset.data_order, "labels": [], "correct": [], "idxs": [],
                   "rank": multiprocess[0], "total": multiprocess[1]}
    # due to the length of sentence there only trainable for batch size of 1
    iter_begin = time.time()
    data_span = len(dataset) // multiprocess[1]
    data_start = multiprocess[0] * data_span
    data_end = (multiprocess[0] + 1) * data_span if multiprocess[0] != multiprocess[1] - 1 else len(dataset)
    print("Process {} handling data from {} to {}".format(multiprocess[0], data_start, data_end))
    
    # Try to load checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path)
    if checkpoint_data is not None and args.resume:
        # Restore state
        assert checkpoint_data["iter_count"] == data_start + len(checkpoint_data["saving_data"]["labels"]) - 1
        saved_iter_count = checkpoint_data["iter_count"]
        iter_count_start = saved_iter_count + 1  # Start from the next iter
        # Ensure iter_count_start does not exceed data_end
        if iter_count_start >= data_end:
            print(f"Checkpoint iter_count {saved_iter_count} is already completed. Starting from beginning.")
            iter_count_start = data_start
        else:
            saving_data = checkpoint_data["saving_data"]
            test_1_acc = checkpoint_data["test_1_acc"]
            test_5_acc = checkpoint_data["test_5_acc"]
            test_count = checkpoint_data["test_count"]
            skip_count = checkpoint_data["skip_count"]
            print(f"Resuming from iter_count={iter_count_start}")
    else:
        iter_count_start = data_start
        print("Starting from beginning")

    for iter_count in range(iter_count_start, data_end):
        user_id, graph, graph_nodes, traj, time_stamps, next_visit, next_time, last_traj_ptr = dataset[iter_count]
        remain_nodes_index = [i for i in range(graph_nodes.shape[0])]
        nodes_label = [-1 for _ in graph_nodes.tolist()]
        recall_count, loop_skip_count = 0, 0  # loop_skip_count is used to count the number of times the loop is skipped
        extra_hint_nodes = None
        pbar = tqdm(range(graph_nodes.shape[0]//args.num_samples + 1), desc=f"Testing:", ncols=100)
        for idx in pbar:
            if idx==0:  # test with no important nodes
                prompt, poi_set, _ = poi_augment(poi_base, user_id, graph, graph_nodes,
                                                [], traj, time_stamps, next_time, last_traj_ptr)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                ).to(model.device)

                pred_visit = testify(model, tokenizer, inputs, args)  # testify with no important nodes
                test_correct = (next_visit == pred_visit[0]) if len(pred_visit) > 0 else False
                if test_correct:
                    for i in range(len(nodes_label)):
                        nodes_label[i] = 0
                    skip_final_test = True
                    break
                if next_visit not in poi_set:
                    extra_hint_nodes = extra_hint(graph, graph_nodes.tolist(), next_visit)
                    print(f"extra hint nodes: {extra_hint_nodes}")
                    if extra_hint_nodes is None:
                        skip_final_test = True
                        break
        ################# iteration for important nodes selection #################
            top_important = args.num_samples if len(remain_nodes_index) > args.num_samples \
                                                   else len(remain_nodes_index)
            imp_nodes = remain_nodes_index[:top_important]
            if not judge_have_extra_hint(extra_hint_nodes, imp_nodes):
                imp_nodes.append(extra_hint_nodes[0])
            
            proj_imp_nodes = [int(graph_nodes[i]) for i in imp_nodes]

            # Prompt Augmentation
            prompt, poi_set, description_pieces = poi_augment(poi_base, user_id, graph, graph_nodes,
                                                                  imp_nodes, traj, time_stamps, next_time, last_traj_ptr)
            # screen the concerned token
            inputs, screen_position = process_tokenize(tokenizer, prompt, description_pieces,
                                                       str(next_visit), model_engine.device)
            # extract embeddings and make them trainable
            input_embeddings, labels, screen_tensor = process_embeddings(model, inputs["input_ids"], screen_position)
            # print("token_length:"+str(inputs["input_ids"].shape[-1]))
            # if inputs["input_ids"].shape[-1] >= 3000:  # an experiential limit for maximum training
            #     # print to the output stream
            #     CONFIG.important_nodes = CONFIG.important_nodes - 1
            #     continue
            acc_1 += int(next_visit in proj_imp_nodes)
            acc_2 += int(next_visit in poi_set)

            # input_embeddings = init_linear(input_embeddings)
            # generate output
            try:
                outputs = model_engine(inputs_embeds=input_embeddings, labels=labels,
                                       attention_mask=inputs['attention_mask'], fp16=True, output_hidden_states=True)
                # get readable output
                text_output = readable_output(outputs, tokenizer, true_mask=screen_position["label_mask"])
                iter_pred_visit = extract_answer(text_output)[0]
                correct = (iter_pred_visit == next_visit)
            except RuntimeError as e:
                print("RuntimeError:", e)
                for i in imp_nodes:
                    nodes_label[i] = 0
                skip_count += 1  # no need to

                if "CUDA out of memory" in str(e):
                    loop_skip_count += 1
                    if loop_skip_count > 5:
                        print("skip too many times, break the loop and save checkpoint")
                        break
                torch.cuda.empty_cache()
                continue
            # DO BACKWARD
            model_engine.zero_grad()
            target_token = inputs["input_ids"][:, plus(screen_position["label_mask"])].flatten()
            pred_logits = outputs.logits[0, screen_position["label_mask"]]  # select useful target

            logp = F.log_softmax(pred_logits.float(), dim=-1)  # Numerical stability
            target_loss = -logp[torch.arange(len(target_token)), target_token].sum()
            if correct:  # only need the true loss grad
                target_loss.backward()
                all_weights_true = cal_all_weight(screen_tensor=screen_tensor)
                # true_outliers = mark_outliers(all_weights_true, k=1.8)
                screen_zero_grad(screen_tensor)  # remove grad
                edge_judger = all_weights_true
            else:  # need to compute both true and false grad
                target_loss.backward(retain_graph=True)  # retain graph for false grad
                all_weights_true = cal_all_weight(screen_tensor=screen_tensor)
                # true_outliers = mark_outliers(all_weights_true, k=1.8)
                screen_zero_grad(screen_tensor)  # remove grad

                # calculate false grad
                model_engine.zero_grad()
                logp = F.log_softmax(pred_logits.float(), dim=-1)  # Numerical stability
                max_token = torch.argmax(pred_logits, dim=-1)
                max_loss = -logp[torch.arange(len(max_token)), max_token].sum()
                max_loss.backward()

                all_weights_false = cal_all_weight(screen_tensor=screen_tensor)
                # false_outliers = mark_outliers(all_weights_false, k=1.8)

                minus_weight = [i-j for i, j in zip(all_weights_true, all_weights_false)]
                # minus_outliers = mark_outliers(minus_weight, k=1.8)

                torch.cuda.empty_cache()
                edge_judger = minus_weight

            ########################### mark ###########################
            # iter_target = [acc_record[checker-1]["target"][i] for i in imp_nodes]
            pos_marks = mark_outliers(edge_judger, k=args.outliner)
            neg_marks = mark_inliners(edge_judger, k=1)
            ########################### testify ###########################
            if sum(pos_marks) > 0:
                prompt, poi_set, description_pieces = poi_augment(poi_base, user_id, graph, graph_nodes,
                                                                [i for i, mark in zip(imp_nodes, pos_marks) if mark],
                                                                traj, time_stamps, next_time, last_traj_ptr)
                # screen the concerned token
                # inputs, screen_position = process_tokenize(tokenizer, prompt, description_pieces,
                #                                            str(next_visit), model_engine.device)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                ).to(model_engine.device)
                pred_visit = testify(model, tokenizer, inputs, args)  # testify with positive nodes
                test_correct = (next_visit in pred_visit)
                recall_count += int(next_visit in poi_set)
                torch.cuda.empty_cache()

            ########################### testify end ###########################
            if sum(pos_marks) > 0 and not test_correct:  # if the test result is wrong, then ignore this result
                for i in range(len(pos_marks)):
                    nodes_label[imp_nodes[i]] = 0
            else:
                for i, (pos, neg) in enumerate(zip(pos_marks, neg_marks)):
                    if pos:
                        nodes_label[imp_nodes[i]] = 1
                    elif neg:
                        nodes_label[imp_nodes[i]] = 0
            remain_nodes_index = [e for e in remain_nodes_index if nodes_label[e] == -1]  # renew index
            pbar.set_postfix({"correct": f"{sum([i==1 for i in nodes_label])}",
                              "recall": f"{recall_count}"})
            
            if len(remain_nodes_index) == 0:
                break
            
        ########################### iteration end ###########################
        # print status
        for n in remain_nodes_index:
            nodes_label[n] = 0
        true_index = [i for i, mark in enumerate(nodes_label) if mark == 1]
        # label_show = [i for i, mark in enumerate(graph_record["target"]) if mark == 1]
        # print(f"target nodes: {label_show}")
        print(f"final selected nodes: {true_index}")
        if len(true_index) == 0:
            print("no correct nodes found")

        ######################## testify start #############################
        if not skip_final_test:
            # true_index = list(set(true_index))
            prompt, poi_set, description_pieces = poi_augment(poi_base, user_id, graph, graph_nodes,
                                                                true_index, traj, time_stamps, next_time, last_traj_ptr)
            # screen the concerned token
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            ).to(model_engine.device)
            pred_visit = testify(model, tokenizer, inputs, args)  # testify with true index
            torch.cuda.empty_cache()

        skip_final_test = False  # reset skip_final_test
        if len(pred_visit) > 0:
            test_1_acc += (next_visit == pred_visit[0])
            test_5_acc += (next_visit in pred_visit)
        test_count += 1
        past_time = time.time() - iter_begin
        print(f"[TESTIFY!!!] test top-1 acc {test_1_acc}/{test_count}={test_1_acc/test_count:.4f}, top-5 acc {test_5_acc}/{test_count}={test_5_acc/test_count:.4f}")
        print("minutes passed: {:.2f}, "
              "estimated minutes left: {:.2f}".format(past_time / 60,
                                                      past_time / (iter_count - iter_count_start + 1) *
                                                      (data_end - (iter_count + 1)) / 60))
        ######################## testify end #############################
        # print result
        saving_data["labels"].append(nodes_label)
        saving_data["correct"].append((next_visit in pred_visit))
        saving_data["idxs"].append(iter_count)

        elapsed = time.time() - iter_begin
        e_hours, e_remainder = divmod(elapsed, 3600)
        e_minutes, e_seconds = divmod(e_remainder, 60)
        predict_time = elapsed / (iter_count - data_start + 1) * (data_end - (iter_count + 1))
        p_hours, p_remainder = divmod(predict_time, 3600)
        p_minutes, p_seconds = divmod(p_remainder, 60)
        content_to_show = {
            "token_size": "{:.4f}".format(inputs["input_ids"].shape[-1]),
            "skip": "{}".format(skip_count),
            "top-1 acc": f"{test_1_acc}/{test_count}={test_1_acc/test_count:.4f}",
            "top-5 acc": f"{test_5_acc}/{test_count}={test_5_acc/test_count:.4f}",
            "ans_time": "{:.1f}".format(time.time() - iter_begin),
            "elapsed": f"{int(e_hours)}:{int(e_minutes)}",
            "time_left": f"{int(p_hours)}:{int(p_minutes)}"
        }

        print("[PROGRESS {}/{}] ".format(iter_count-data_start, data_end-data_start) + json.dumps(content_to_show), flush=True)  # print to the output stream

        # Save checkpoint after every 100 loops
        if len(saving_data["labels"]) % 10 == 0:
            save_checkpoint(checkpoint_path, iter_count, saving_data,
                            test_1_acc, test_5_acc, test_count, skip_count)
    
    content = {'model': 'llm', 'tag': "begin", 'total_steps': '{}'.format(len(dataset))}
    print("[PROGRESS] " + json.dumps(content), flush=True)

    if multiprocess[1] <= 1:
        file_name = "all_labels.json"
    else:
        file_name = "label_result_{args.mode}_{}_{}.json".format(args.mode, multiprocess[0], multiprocess[1])
    with open(f"./data/GA_labels/{dataset_name}_2/{model_name.split('/')[-1]}/{file_name}", "w", encoding="utf-8") as f:
        json.dump(saving_data, f, ensure_ascii=False, indent=4)
        print("successfully saved final result.")
    
    # # Delete checkpoint after program completion
    # if os.path.exists(checkpoint_path):
    #     os.remove(checkpoint_path)
    #     print("Checkpoint file removed after completion.")
    
    return


if __name__ == "__main__":
    # deepspeed args
    parser = argparse.ArgumentParser(description='Description of your script')
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--p_rank", type=int, default=1)
    parser.add_argument("--p_total", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B",
                        choices=["Qwen/Qwen3-4B", "meta-llama/Meta-Llama-3-8B"])
    parser.add_argument("--dataset", type=str, default="nyc",
                        choices=["nyc", "tky", "cal", "flo"])
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "valid", "test"])
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--outliner", type=float, default=1.8)
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint", default=True)
    args = parser.parse_args()
    print("[LLM] CUDA_VISIBLE_DEVICES: ", os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"))
    CONFIG = init(args.dataset)
    # model and data paths

    # init paths
    poi_path = os.path.join("./data/{}_pp".format(args.dataset), "poi_points.csv")
    pp_path = os.path.join("./data/{}_pp".format(args.dataset), "pped_v2.pkl")
    llm_process(args, pp_path, poi_path)
    print("LLM finished")
