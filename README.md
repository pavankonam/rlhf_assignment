# RLHF Assignment: Reinforcement Learning from Human Feedback

Implementation of PPO, GRPO, and DPO for aligning language models using the Anthropic HH-RLHF dataset with GPT-2 (124M parameters).

This repository implements:

- **Part 1** – Preference data exploration, preprocessing, and reward model training  
- **Part 2** – Policy optimization with **PPO** and **GRPO**  
- **Part 3** – **DPO** as a direct preference optimization baseline  
- **Part 4** – Quantitative and qualitative evaluation, including GPT-4-as-judge win rates, reward distributions, KL drift, and failure mode analysis

All reported results (tables, figures, and checkpoints) in `ANALYSIS.md` were generated from this codebase.

---

## Project Structure

```bash
rlhf_assignment/
├── ANALYSIS.md
├── Dockerfile               # Containerized environment (GPU-accelerated)
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── run_all.sh               # Convenience script to run full pipeline
├── outputs/
│   ├── dataset_stats.json   # Dataset-level statistics (lengths, filters, etc.)
│   ├── evaluation/          # Quantitative & qualitative evaluation artifacts
│   ├── figures/             # Plots (training curves, dataset analysis, Pareto)
│   ├── models/              # Trained checkpoints for reward, PPO, GRPO, DPO
│   ├── processed_data/      # Tokenized train/val/test splits + metadata
│   └── samples/             # Additional sample generations (if any)
├── scripts/
│   ├── step1_data_exploration.py    # Part 1.1 – dataset analysis
│   ├── step1b_preprocess.py         # Part 1.1 – preprocessing pipeline
│   ├── step2_train_reward_model.py  # Part 1.2 – reward model training
│   ├── step3_train_ppo.py           # Part 2.1 – PPO training
│   ├── step4_train_grpo.py          # Part 2.2 – GRPO training
│   ├── step5_train_dpo.py           # Part 3   – DPO training
│   └── step6_evaluate.py            # Part 4   – evaluation & analysis
├── src/
│   ├── __init__.py
│   ├── data/                        # (Reserved) dataset utilities
│   ├── evaluation/                  # (Reserved) evaluation helpers
│   ├── models/
│   │   ├── __init__.py
│   │   └── reward_model.py          # Reward model architecture
│   └── trainers/                    # (Reserved) training helpers (if extended)
└── venv/                            # Local virtualenv (not used in Docker)
```

---

## Quick Start

You can run this project in three ways:

1. **Docker (recommended, for reproducible grading)**
2. **Local virtual environment (Python 3.9+)**
3. **GPU server (e.g., DeepDish) with CUDA-enabled PyTorch**

---

## 1. Running with Docker (Recommended)

### 1.1 Build the image

From the root of the repository:

```bash
docker build -t rlhf-assignment .
```

### 1.2 Run full pipeline (end-to-end)

This will:

- Download the Anthropic HH-RLHF dataset
- Preprocess and tokenize data
- Train the reward model
- Train PPO, GRPO, and DPO policies
- Run evaluation (reward distributions, win rates, KL, failure analysis)
- Save all artifacts in `outputs/`

```bash
docker run --gpus all   -e OPENAI_API_KEY="your-openai-api-key"   -v $(pwd)/outputs:/app/outputs   rlhf-assignment
```

The `CMD` in the Dockerfile calls `run_all.sh`, which sequentially executes:

```bash
python scripts/step1_data_exploration.py
python scripts/step1b_preprocess.py
python scripts/step2_train_reward_model.py
python scripts/step3_train_ppo.py
python scripts/step4_train_grpo.py
python scripts/step5_train_dpo.py
python scripts/step6_evaluate.py
```

### 1.3 Run interactively (step-by-step inside container)

```bash
docker run --gpus all -it   -e OPENAI_API_KEY="your-openai-api-key"   -v $(pwd)/outputs:/app/outputs   rlhf-assignment bash
```

Then inside the container:

```bash
python scripts/step1_data_exploration.py
python scripts/step1b_preprocess.py
python scripts/step2_train_reward_model.py
python scripts/step3_train_ppo.py
python scripts/step4_train_grpo.py
python scripts/step5_train_dpo.py
python scripts/step6_evaluate.py
```

---

## 2. Running Locally (Without Docker)

### 2.1 Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 2.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 Set required environment variables

```bash
export TOKENIZERS_PARALLELISM=false
export OPENAI_API_KEY="your-openai-api-key"  # For GPT-4-as-judge evaluation
```

### 2.4 Run the full pipeline

```bash
bash run_all.sh
```

Or run step-by-step (recommended during development):

```bash
python scripts/step1_data_exploration.py
python scripts/step1b_preprocess.py
python scripts/step2_train_reward_model.py
python scripts/step3_train_ppo.py
python scripts/step4_train_grpo.py
python scripts/step5_train_dpo.py
python scripts/step6_evaluate.py
```

---

## 3. Running on a GPU Server (e.g., DeepDish)

On a CUDA-enabled machine (like Northwestern DeepDish):

```bash
cd rlhf_assignment

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

export TOKENIZERS_PARALLELISM=false
export OPENAI_API_KEY="your-openai-api-key"
```

Use a specific GPU if needed:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/step1_data_exploration.py
CUDA_VISIBLE_DEVICES=0 python scripts/step1b_preprocess.py
CUDA_VISIBLE_DEVICES=0 python scripts/step2_train_reward_model.py
CUDA_VISIBLE_DEVICES=0 python scripts/step3_train_ppo.py
CUDA_VISIBLE_DEVICES=0 python scripts/step4_train_grpo.py
CUDA_VISIBLE_DEVICES=0 python scripts/step5_train_dpo.py
CUDA_VISIBLE_DEVICES=0 python scripts/step6_evaluate.py
```

---

## Compute Requirements

### Minimum Requirements

- **Python**: 3.9+
- **CPU RAM**: 16 GB
- **Disk**: ~15 GB free (dataset + checkpoints + outputs)
- **GPU**: 8 GB VRAM (bare minimum; will require smaller batch sizes and longer training)

### Recommended Setup (used for reported results)

The reported results in `ANALYSIS.md` were produced with:

- **GPU**: NVIDIA **RTX A6000**, 48 GB VRAM  
- **Peak GPU Memory Usage**: ~17.3 GB  
- **CPU RAM**: 32 GB  
- **Disk Space Used**: ~25 GB  
- **Total Training Time**: ≈ **6.5 hours**

This setup easily supports the default batch sizes and sequence lengths.

---

## Configuration Overview

Key hyperparameters are stored in Python `CONFIG` dictionaries within each script:

### Reward Model – `scripts/step2_train_reward_model.py`

```python
CONFIG = {
    "model_name": "gpt2",
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
}
```

- Backbone: GPT-2 (124M parameters)
- Reward head: small MLP on top of pooled hidden states
- Loss: Pairwise ranking loss over chosen vs rejected responses

### PPO – `scripts/step3_train_ppo.py`

```python
CONFIG = {
    "clip_ratio": 0.2,          # PPO clipping (ε)
    "kl_coef": 0.1,             # KL penalty coefficient (β)
    "entropy_coef": 0.01,       # Entropy bonus weight
    "learning_rate": 1e-5,
    "ppo_epochs": 4,
    "num_train_prompts": 10000,
    "num_eval_prompts": 500,
}
```

- **Advantages**: computed as `rewards - value_baseline`, where the baseline is the value head of the policy model, then **normalized per batch**:
  - `advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)`
- **Returns**: set to the scalar reward for each generated response (no temporal discounting).

### GRPO – `scripts/step4_train_grpo.py`

```python
CONFIG = {
    "group_size": 4,            # G: responses per prompt (4–8 as in spec)
    "learning_rate": 1e-5,
    "num_epochs": 2,
    "kl_coef": 0.1,
    "num_train_prompts": 10000,
    "num_eval_prompts": 500,
}
```

- For each prompt, **G=4** responses are sampled.
- Group-relative advantages are computed as:
  - \(A_i = (r_i - \mu_{group}) / (\sigma_{group} + 1e-8)\)
- Uses a simplified policy gradient without clipping or a separate value function.

### DPO – `scripts/step5_train_dpo.py`

```python
CONFIG = {
    "beta": 0.1,                # DPO inverse temperature
    "learning_rate": 5e-7,
    "num_epochs": 1,
    "batch_size": 4,
}
```

- Directly optimizes preferences using the DPO objective, without training an explicit reward model.

---

## Dataset

We use the **Anthropic HH-RLHF** dataset from Hugging Face:

- Dataset: `Anthropic/hh-rlhf`
- Data contains:
  - Prompts
  - Pairs of responses: `(chosen, rejected)` based on human preference
- The preprocessing script filters:
  - Ties / identical responses
  - Extremely short responses
  - Truncates to `max_length = 512` tokens

Preprocessed splits (after filtering):

- Train: 139,505 examples  
- Validation: 15,501 examples  
- Test: 8,231 examples  

The `outputs/processed_data/` directory contains tokenized tensors and metadata JSON files.

---

## Models

| Model           | Parameters | Description                                   |
|----------------|-----------:|-----------------------------------------------|
| Base Policy    | 124M       | Pretrained GPT-2                              |
| Reward Model   | 124M + head| GPT-2 + scalar reward head trained on prefs   |
| PPO Policy     | 124M       | Base policy fine-tuned via PPO                |
| GRPO Policy    | 124M       | Base policy fine-tuned via GRPO               |
| DPO Policy     | 124M       | Base policy fine-tuned via DPO (no reward net)|

All trained checkpoints are saved under `outputs/models/…`.

---

## Outputs

After a full run (`bash run_all.sh` or Docker default `CMD`), the following artifacts are produced:

```bash
outputs/
├── dataset_stats.json                # Dataset length histograms & filter stats
├── processed_data/
│   ├── train_tokenized.pt
│   ├── val_tokenized.pt
│   ├── test_tokenized.pt
│   ├── train_metadata.json
│   ├── val_metadata.json
│   ├── test_metadata.json
│   └── config.json
├── models/
│   ├── reward_model/
│   │   ├── best_model.pt
│   │   ├── training_history.json
│   │   ├── error_analysis.json
│   │   └── reward_model_training_curves.png
│   ├── ppo/
│   │   ├── checkpoint_epoch1.pt
│   │   ├── checkpoint_epoch2.pt
│   │   ├── final_model.pt
│   │   ├── generated_samples.json
│   │   ├── training_history.json
│   │   └── training_curves.png
│   ├── grpo/
│   │   ├── checkpoint_epoch1.pt
│   │   ├── checkpoint_epoch2.pt
│   │   ├── final_model.pt
│   │   ├── generated_samples.json
│   │   ├── training_history.json
│   │   └── training_curves.png
│   └── dpo/
│       ├── checkpoint_epoch1.pt
│       ├── final_model.pt
│       ├── generated_samples.json
│       ├── training_history.json
│       └── training_curves.png
├── evaluation/
│   ├── evaluation_summary.json       # Aggregated metrics for all models
│   ├── pareto_comparison.json        # Reward vs KL Pareto data
│   ├── win_rates.json                # GPT-4-as-judge win rates
│   ├── failure_analysis.json         # Adversarial prompt evaluation
│   ├── generated_samples_all.json    # 20 samples per model
│   ├── pareto_frontier.png
│   └── reward_distributions.png
└── figures/
    └── dataset_analysis.png          # Dataset statistics (lengths, filters)
```

---

## Results Summary (High Level)

For full details, see **`ANALYSIS.md`**. Briefly:

- Reward model reaches **~62.8%** validation accuracy on preference prediction.
- PPO and GRPO improve reward scores but sometimes underperform the base model in head-to-head comparisons, indicating partial reward hacking and limitations from a ~62%-accurate reward model.
- **DPO** achieves the best overall alignment:
  - Highest mean reward under the reward model
  - Best GPT-4-as-judge win rates vs all other models

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch sizes in the respective `CONFIG` dictionaries, e.g.:

```python
"batch_size": 4
```

or lower `max_length` during development.

### Tokenizer Parallelism Warnings

Export:

```bash
export TOKENIZERS_PARALLELISM=false
```

### OpenAI API Issues

- Ensure `OPENAI_API_KEY` is set in the environment.
- Step 6 (evaluation) will skip GPT-4 judging if the key is missing or invalid, but other metrics (reward scores, KL, etc.) will still be computed.

---

## Reproducibility Notes

- The fixed random seeds are set inside the training scripts where applicable.
- All hyperparameters, compute details, and dataset statistics used in the report are derived from the run that produced the committed `outputs/` directory.
- The **analysis in `ANALYSIS.md` directly corresponds to those artifacts**, so a grader can replicate or inspect every table and figure from saved JSON and PNG files.
