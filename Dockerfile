# RLHF Assignment Dockerfile
# Supports PPO, GRPO, and DPO training with GPT-2 on Anthropic HH-RLHF dataset

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TOKENIZERS_PARALLELISM=false

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy base project files
COPY requirements.txt /app/requirements.txt
COPY run_all.sh /app/run_all.sh
COPY scripts /app/scripts
COPY src /app/src
COPY ANALYSIS.md /app/ANALYSIS.md
COPY README.md /app/README.md

# Create outputs directory (will be mounted as a volume)
RUN mkdir -p /app/outputs

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Make run_all executable
RUN chmod +x /app/run_all.sh

# Default CMD prints instructions and runs the full pipeline
CMD ["/bin/bash", "-lc", "echo 'RLHF Assignment Container' && \
    echo '=================================' && \
    echo 'To run end-to-end training & evaluation:' && \
    echo '  docker run --gpus all -e OPENAI_API_KEY=... -v $(pwd)/outputs:/app/outputs <image>' && \
    echo '' && \
    echo 'This executes:' && \
    echo '  bash run_all.sh' && \
    echo '' && \
    echo 'Inside container, run steps individually:' && \
    echo '  python scripts/step1_data_exploration.py' && \
    echo '  python scripts/step1b_preprocess.py' && \
    echo '  python scripts/step2_train_reward_model.py' && \
    echo '  python scripts/step3_train_ppo.py' && \
    echo '  python scripts/step4_train_grpo.py' && \
    echo '  python scripts/step5_train_dpo.py' && \
    echo '  python scripts/step6_evaluate.py' && \
    echo '' && \
    bash run_all.sh"]
