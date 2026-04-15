#!/bin/bash
# Deploy and run LoRA experiments on Aliyun A10 server
# Usage: bash scripts/deploy_aliyun.sh

SERVER="root@39.105.43.10"
KEY="$HOME/.ssh/aliyun123.pem"
REMOTE_DIR="/root/multi-agent-train"

echo "=== Uploading code ==="
cd ~/auto-research/multi-agent-train
tar czf /tmp/multi-agent-train.tar.gz \
  --exclude='.git' --exclude='paper/*.pdf' --exclude='checkpoints' \
  --exclude='runs' --exclude='logs' --exclude='*.pdf' --exclude='*.aux' \
  --exclude='*.log' --exclude='*.out' --exclude='results/individual' \
  -C ~/auto-research multi-agent-train

scp -i $KEY /tmp/multi-agent-train.tar.gz $SERVER:/root/
ssh -i $KEY $SERVER "cd /root && tar xzf multi-agent-train.tar.gz"

echo "=== Starting LoRA experiments on Aliyun A10 ==="
# Run LoRA comparison with Qwen on Aliyun
ssh -i $KEY $SERVER "export PATH=/root/miniconda3/bin:\$PATH && \
  cd $REMOTE_DIR && \
  mkdir -p results logs && \
  nohup bash -c 'PYTHONPATH=. python scripts/compare_lora.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --episodes 10000 \
    --seeds 42 123 456 \
    --lora-rank 8 \
    --comm-methods vq_ssr ssr_v2 discrete \
    --output results/lora_comparison.json \
    --device cuda' > logs/lora_comparison.log 2>&1 &"

echo "=== LoRA experiments launched on Aliyun ==="
echo "Monitor with: ssh -i $KEY $SERVER 'tail -f $REMOTE_DIR/logs/lora_comparison.log'"
