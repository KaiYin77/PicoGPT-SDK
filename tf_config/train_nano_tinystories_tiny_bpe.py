"""
TensorFlow training configuration for NanoGPT on TinyStories dataset
2-layer transformer for better quality while staying compact

Target: ~300-350K parameters for efficient INT8 quantization
CIMv3-compatible architecture
Dataset: TinyStories (simple English stories, no Unicode issues)
"""

# I/O
out_dir = 'out-tf-nano-tinystories-tiny-bpe'
eval_interval = 600   # Adjusted for training length
log_interval = 10
eval_iters = 200
always_save_checkpoint = True

# Data
dataset = 'tinystories_tiny_bpe'
# TinyStories with BPE produces clean tokens; larger batch for stable training
batch_size = 128
block_size = 128  # Context length (can increase to 256 if memory allows)

# CIMv3-Compatible Nano GPT architecture
# CIMv3 constraints: n_embd must be 128/256/512, d_head should be 64
n_layer = 2      # 2 transformer layers for better quality (vs 1-layer SUB-Pico)
n_head = 2       # 2 attention heads (128 / 2 = 64 d_head - CIMv3 optimized)
n_embd = 128     # 128 embedding dimensions (CIMv3 compatible)
dropout = 0.1    # Light dropout for nano model
bias = False     # CIMv3 prefers bias-free INT8 GEMM operations

# AdamW optimizer
learning_rate = 1.5e-3  # Slightly lower than 1-layer (deeper model needs more stability)
max_iters = 35000       # Same as SUB-Pico (adequate for 200K samples)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Learning rate decay
warmup_iters = 1500     # Longer warmup for 2-layer model (vs 1000 for 1-layer)
lr_decay_iters = 35000  # Match max_iters
min_lr = 1.5e-4

# Mixed precision (TensorFlow equivalent of PyTorch AMP)
# NOTE: Set to False for CPU training - mixed precision is slower on CPU
mixed_precision = False  # Use 'mixed_float16' policy (GPU only)

# Dataset-specific notes:
# TinyStories advantages:
#   - Clean ASCII text (no Unicode encoding issues)
#   - Simple vocabulary (~1,500 words vs WikiText's 30,000+)
#   - Coherent narrative structure
#   - Perfect for 256-vocab BPE tokenizer
#
# Expected performance compared to SUB-Pico (1-layer):
#   - Better quality: 2 layers capture more complex patterns
#   - Better coherence: longer-range dependencies
#   - Slightly slower training: ~2x compute per iteration
#   - Better final loss: deeper model converges to better minimum

# CIMv3-Compatible Model size estimation:
# Input embedding: vocab_size * n_embd = 256 * 128 = 32,768
# Position embedding: block_size * n_embd = 128 * 128 = 16,384
#
# Per transformer layer (CIMv3 sequence: MHA → FFN → LN2, no LN1):
#   - Attention QKV: n_embd * (3 * n_embd) = 128 * 384 = 49,152
#   - Attention proj: n_embd * n_embd = 128 * 128 = 16,384
#   - MLP fc: n_embd * (2 * n_embd) = 128 * 256 = 32,768
#   - MLP proj: (2 * n_embd) * n_embd = 256 * 128 = 32,768
#   - LayerNorm 2: n_embd = 128
#   - Total per layer: 49,152 + 16,384 + 32,768 + 32,768 + 128 = 131,200
#
# 2 transformer layers: 2 * 131,200 = 262,400
# Output projection: tied with input embedding (0 additional params)
# Final LayerNorm: n_embd = 128
#
# Total: 32,768 + 16,384 + 262,400 + 128 = 311,680 parameters
# CIMv3 Features: ReLU activation, no bias, d_head=64 optimized
#
# Comparison:
#   - SUB-Pico (1 layer):  ~180K parameters
#   - Nano (2 layers):     ~311K parameters
#   - Memory efficient: still fits in embedded systems
#   - Quality gain: significant improvement in coherence
#
# For even better quality on TinyStories, consider:
#   - Increasing n_layer to 3: ~442K parameters
#   - Increasing n_embd to 256: ~1.2M parameters (requires more RAM)
#   - Increasing max_iters to 50000 (better convergence for deeper model)
