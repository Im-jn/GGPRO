# PGA_label

This project implements a Point of Interest (POI) recommendation and trajectory prediction system using Large Language Models (LLMs) and Graph Neural Networks (GNNs). It involves fine-tuning LLMs with LoRA and using a gradient-based labeling approach for optimization.

## Features

- **Data Preprocessing**: Converts raw POI check-in data (Foursquare, Weeplaces) into structured datasets.
- **LLM Fine-tuning**: Supports fine-tuning Qwen and Llama 3 models using LoRA and DeepSpeed.
- **Gradient Labeling**: An iterative process to generate labels and optimize the model using gradient information.
- **Hybrid Architecture**: Combines LLMs for sequence understanding and GNNs for spatial/graph-based context.

## Requirements

- Python 3.8+
- PyTorch
- DeepSpeed
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- DGL (Deep Graph Library)
- Pandas, NumPy, Scipy
- NLTK

Install dependencies via pip:

```bash
pip install torch deepspeed transformers peft dgl pandas numpy scipy nltk
```

## Supported Datasets

- `nyc`: Foursquare NYC
- `tky`: Foursquare Tokyo

## Usage

### 1. Data Preparation

Preprocess the raw datasets into the required format.

```bash
python data_pp.py --dataset nyc
```
*Replace `nyc` with `tky` as needed.*

### 2. Fine-tuning LLM

Fine-tune the base LLM on the trajectory data using LoRA.

```bash
python finetune.py --dataset nyc --model_name Qwen/Qwen3-4B
```

### 3. Gradient Labeling / Training

Run the gradient-based labeling and training process. This script uses DeepSpeed for distributed execution.

```bash
deepspeed -p XXXX gradient_label.py --dataset nyc --model_name Qwen/Qwen3-4B --mode train
```

For models supporting Flash Attention 2, use:

```bash
python gradient_label_flash.py --dataset nyc --model_name Qwen/Qwen3-4B --mode train
```

### 4. Evaluation

Test the trained model performance.

```bash
python test.py --dataset nyc --model_name Qwen/Qwen3-4B --mode normal
```

**Modes:**
- `normal`: Standard testing with full context.
- `dummy`: Uses a dummy judger.
- `random`: Uses a random judger.
- `no_hint`: Testing without hint information.

## File Structure

- `data_pp.py`: Data preprocessing script.
- `finetune.py`: Script for fine-tuning LLMs using PEFT/LoRA.
- `gradient_label.py`: Main training loop with gradient-based labeling.
- `test.py`: Evaluation script.
- `database.py`: Dataset and POI graph handling.
- `small_models.py`: GNN and other small model definitions.
- `prompt.py`: Prompt engineering and augmentation logic.
- `config.py`: Configuration management.
- `ds_ft_config.json`: DeepSpeed configuration.

## Configuration

You can adjust hyperparameters such as `top_trans`, `history_visits`, `gnn_in_feat`, etc., in `config.py`. Dataset-specific configurations can be loaded if available.



