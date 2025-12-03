#!/bin/bash
# RLHF Assignment - Full Pipeline Execution Script
# Runs all steps sequentially with error checking

set -e  # Exit on error

echo "=========================================="
echo "RLHF Assignment - Full Pipeline"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Set environment variables
export TOKENIZERS_PARALLELISM=false

# Check for OpenAI API key (needed for evaluation)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set. GPT-4 evaluation will be skipped."
    echo "Set it with: export OPENAI_API_KEY='your-key'"
    echo ""
fi

# Create output directories
mkdir -p outputs/processed_data
mkdir -p outputs/models/reward_model
mkdir -p outputs/models/ppo
mkdir -p outputs/models/grpo
mkdir -p outputs/models/dpo
mkdir -p outputs/evaluation
mkdir -p outputs/figures
mkdir -p outputs/samples

# Step 1: Data Exploration (Part 1.1 Task A)
echo ""
echo "=========================================="
echo "Step 1/7: Data Exploration (Part 1.1 Task A)"
echo "=========================================="
python scripts/step1_data_exploration.py
echo "✓ Step 1 complete"

# Step 1b: Preprocessing (Part 1.1 Task B)
echo ""
echo "=========================================="
echo "Step 2/7: Data Preprocessing (Part 1.1 Task B)"
echo "=========================================="
python scripts/step1b_preprocess.py
echo "✓ Step 2 complete"

# Step 2: Reward Model Training (Part 1.2)
echo ""
echo "=========================================="
echo "Step 3/7: Reward Model Training (Part 1.2)"
echo "=========================================="
python scripts/step2_train_reward_model.py
echo "✓ Step 3 complete"

# Step 3: PPO Training (Part 2.1)
echo ""
echo "=========================================="
echo "Step 4/7: PPO Training (Part 2.1)"
echo "=========================================="
python scripts/step3_train_ppo.py
echo "✓ Step 4 complete"

# Step 4: GRPO Training (Part 2.2)
echo ""
echo "=========================================="
echo "Step 5/7: GRPO Training (Part 2.2)"
echo "=========================================="
python scripts/step4_train_grpo.py
echo "✓ Step 5 complete"

# Step 5: DPO Training (Part 3)
echo ""
echo "=========================================="
echo "Step 6/7: DPO Training (Part 3)"
echo "=========================================="
python scripts/step5_train_dpo.py
echo "✓ Step 6 complete"

# Step 6: Evaluation (Part 4)
echo ""
echo "=========================================="
echo "Step 7/7: Evaluation (Part 4)"
echo "=========================================="
python scripts/step6_evaluate.py
echo "✓ Step 7 complete"

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Outputs saved to:"
echo "  - outputs/processed_data/     : Preprocessed datasets"
echo "  - outputs/models/             : Trained model checkpoints"
echo "  - outputs/evaluation/         : Evaluation results"
echo "  - outputs/figures/            : Training curves and plots"
echo ""
echo "Generated samples (20 per model):"
echo "  - outputs/models/ppo/generated_samples.json"
echo "  - outputs/models/grpo/generated_samples.json"
echo "  - outputs/models/dpo/generated_samples.json"
echo ""
echo "See ANALYSIS.md for detailed results analysis"
echo "=========================================="