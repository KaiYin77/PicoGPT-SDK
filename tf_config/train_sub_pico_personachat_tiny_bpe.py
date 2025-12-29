"""
TensorFlow training configuration for SUB-PicoGPT on PersonaChat dataset
Smaller model for faster training / experimentation with conversational AI

Target: ~300-400K parameters (smaller, faster, but lower quality conversations)
CIMv3-compatible architecture
Dataset: Synthetic-Persona-Chat (persona-based dialogues)
"""

# I/O
out_dir = 'out-tf-sub-pico-personachat-tiny-bpe'
eval_interval = 500  # More frequent checkpoints (smaller dataset)
log_interval = 10
eval_iters = 200
always_save_checkpoint = True

# Data
dataset = 'personachat_tiny_bpe'
# PersonaChat has longer conversations
batch_size = 96       # Medium batch size
block_size = 256      # Longer context for conversations

# SUB-Pico GPT architecture - smaller but still conversational
n_layer = 2       # 2 transformer layers (compromise between speed and quality)
n_head = 2        # 2 attention heads (128 / 2 = 64 d_head)
n_embd = 128      # 128 embedding dimensions (CIMv3 compatible)
dropout = 0.1     # Light dropout
bias = False      # CIMv3 prefers bias-free

# AdamW optimizer
learning_rate = 1.5e-3  # Moderate LR
max_iters = 50000       # Extended for smaller dataset (needs more epochs)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Learning rate decay
warmup_iters = 1000     # Longer warmup for 2-layer model
lr_decay_iters = 50000  # Match max_iters
min_lr = 1.5e-4

# Mixed precision
mixed_precision = False

# Model size estimation (2 layers, 128 embd):
# Input embedding: 512 * 128 = 65,536
# Position embedding: 256 * 128 = 32,768
# Per layer: ~262K params
# 2 layers: ~524K
# Total: ~622K parameters
#
# This is a good balance between:
#   - Small enough to train quickly
#   - Large enough for basic conversations
#   - Better than 1-layer 128-embd (~180K params)
