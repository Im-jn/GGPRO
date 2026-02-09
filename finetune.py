import argparse
import gc
import os
from collections import defaultdict
import random
from dataclasses import dataclass

import deepspeed
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import TrainingArguments, Trainer, PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Optional, Sequence
import copy
# from LLMs import HF_model
from config import CONFIG, print_config, init
from database import POIBase, TrajDataset
from prompt import poi_augment


class HF_dataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, traj_dataset, poi_base, tokenizer):
        super(HF_dataset, self).__init__()
        self.poi_base = poi_base
        self.dataset = traj_dataset
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer.model_max_length = 2048

    def augment(self, user_id, graph, graph_nodes, imp_nodes, traj, time_stamps, next_time):
        prompt, _, _ = poi_augment(self.poi_base, user_id, graph, graph_nodes, imp_nodes, traj, time_stamps, next_time, 
                                   with_answer_instruction=False)
        return prompt

    def preprocess(self, question, next_time, next_visit):
        answer = "<Answer>: At {}, the user will visit POI id {}.{}" .format(
            next_time.strftime('%Y-%m-%d %H:%M:%S'), next_visit, self.tokenizer.eos_token)

        Q_and_A = self.tokenizer(question + answer, return_tensors="pt", padding="longest",
                                 max_length=self.tokenizer.model_max_length, truncation=True)
        if Q_and_A["input_ids"].shape[-1] > 2000:  # if too long
            question = question[-4000:]
            Q_and_A = self.tokenizer(question + answer, return_tensors="pt", padding="longest",
                                     max_length=self.tokenizer.model_max_length, truncation=True)

        Q_len = len(self.tokenizer(question, return_tensors="pt", padding="longest",
                                   max_length=self.tokenizer.model_max_length, truncation=True)["input_ids"][0])

        input = Q_and_A["input_ids"].flatten()
        label = copy.deepcopy(input)
        label[:Q_len] = -100
        return input, label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        current_data = self.dataset[i]
        user_id, graph, graph_nodes, traj, time_stamps, next_visit, next_time, last_traj_ptr = current_data
        select_num = CONFIG.important_nodes if CONFIG.important_nodes < len(graph_nodes) else len(graph_nodes)
        important_nodes = random.sample(list(range(len(graph_nodes))), select_num)
        question = self.augment(user_id, graph, graph_nodes, important_nodes, traj, time_stamps, next_time)
        input, label = self.preprocess(question, next_time, next_visit)
        return dict(input_ids=input, labels=label)


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class MyTrainer(Trainer):
    def get_train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            shuffle=False  # shuffle and sampler are mutually exclusive, generally turn off shuffle
        )


def HF_finetune(datasets, poi_base):
    # Set LoRA configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model_name = ["Qwen/Qwen2.5-0.5B", "meta-llama/Meta-Llama-3-8B"][1]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Ensure model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Convert model to PEFT supported model
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()  # Check number of trainable parameters

    training_args = TrainingArguments(
        output_dir=CONFIG.data_out_dir,
        # evaluation_strategy="epoch",
        bf16=True,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=CONFIG.data_out_dir+"/logs",
        save_total_limit=1,
        save_strategy="epoch",
        dataloader_drop_last=True,
    )
    data_collator = DataCollator(tokenizer)

    trainer = MyTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=HF_dataset(datasets["train"], poi_base, tokenizer),
        eval_dataset=HF_dataset(datasets["valid"], poi_base, tokenizer),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir=os.path.join(training_args.output_dir, model_name+"/model"))
    try:
        trainer.save_state()
    except:
        print("use alternate")
        trainer.save_state(trainer)
    print("\nsuccessfully saved model.")


def HF_finetune_v2(model_name, datasets, poi_base):
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Ensure model is on the correct device (DeepSpeed handles this automatically, but explicit setting is safer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Apply LoRA
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(CONFIG.data_out_dir, "checkpoints"),
        deepspeed="ds_ft_config.json",  # ⭐ Add DeepSpeed config
        fp16=True,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=1,
        save_strategy="epoch",
        logging_dir=os.path.join(CONFIG.data_out_dir, "logs"),
        dataloader_drop_last=True,
        logging_steps=500,
        # max_steps=10000,
        report_to="none",  # Disable wandb etc. upload
    )

    data_collator = DataCollator(tokenizer)

    trainer = MyTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=HF_dataset(datasets["train"], poi_base, tokenizer),
        eval_dataset=HF_dataset(datasets["valid"], poi_base, tokenizer),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir=os.path.join(CONFIG.data_out_dir,
                                               model_name.split("/")[-1] + "/model"))
    peft_model.save_pretrained(os.path.join(CONFIG.data_out_dir,
                                            model_name.split("/")[-1] + "/pretrained"))
    try:
        trainer.save_state()
    except:
        trainer.save_state(trainer)

    print("\n✅ Successfully saved model to {}".format(os.path.join(training_args.output_dir, model_name.replace("/", "_"))))


def resave_pretrained():
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model_name = ["Qwen/Qwen3-4B", "meta-llama/Meta-Llama-3-8B"][0]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Apply LoRA
    peft_model = get_peft_model(model, peft_config)
    peft_model.load_state_dict(torch.load(os.path.join(CONFIG.data_out_dir,
                                                       model_name.split("/")[-1], "model", "training_args.bin")))
    peft_model.save_pretrained(os.path.join(CONFIG.data_out_dir, model_name.split("/")[-1] + "/pretrained"))
    print("saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B",
                        choices=["Qwen/Qwen3-4B", "meta-llama/Meta-Llama-3-8B"])
    parser.add_argument("--dataset", type=str, default="nyc",
                        choices=["nyc", "tky", "cal", "flo"])
    args = parser.parse_args()
    CONFIG = init(args.dataset)

    # init paths
    CONFIG.dataset_name = args.dataset
    CONFIG.data_pp_dir = "./data/{}_pp".format(CONFIG.dataset_name)
    CONFIG.data_out_dir = "./data/{}_out".format(CONFIG.dataset_name)
    CONFIG.peft_model_dir = "./data/{}_out".format(CONFIG.dataset_name)
    print_config(CONFIG)
    poi_path = os.path.join("./data/{}_pp".format(args.dataset), "poi_points.csv")
    pp_path = os.path.join("./data/{}_pp".format(args.dataset), "pped_v2.pkl")

    poi_base = POIBase(poi_path)
    print("There are {} pois and {} categories".format(len(poi_base.pois_df), len(poi_base.cate_list)))

    datasets = {"train": TrajDataset(pp_path=pp_path, action="train", shuffle=CONFIG.shuffle),
                "valid": TrajDataset(pp_path=pp_path, action="valid", shuffle=CONFIG.shuffle)}
    print("Loaded datasets: train size = {}, valid size = {}".format(len(datasets["train"]), len(datasets["valid"])))

    HF_finetune_v2(args.model_name, datasets, poi_base)
