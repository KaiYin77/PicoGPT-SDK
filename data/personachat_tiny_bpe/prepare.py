"""
Prepare the PersonaChat dataset with a tiny BPE tokenizer for conversational AI.
Saves train.bin, val.bin containing token ids, and meta.pkl with tokenizer info.

Dataset: https://huggingface.co/datasets/google/Synthetic-Persona-Chat
Version: Synthetic-Persona-Chat (extended with synthetic conversations)

Synthetic-Persona-Chat is a high-quality persona-based dialogue dataset
designed for training conversational AI models with personality awareness.
"""
import os
import pickle
import numpy as np
from collections import defaultdict
import time

# Special conversation tokens
SPECIAL_TOKENS = {
    '<U1>': 0,    # User 1 turn marker
    '<U2>': 1,    # User 2 turn marker
    '<END>': 2,   # End of conversation
    '<PERSONA>': 3,  # Persona description marker (optional)
}

# BPE config
TARGET_VOCAB_SIZE = 512  # Larger vocab for natural conversation (was 256 for stories)
SP_MARKER = "\u2581"  # SentencePiece-style space marker

# Dataset config
MAX_TRAIN_SAMPLES = None   # Use all training data (~8,944 conversations)
MAX_VAL_SAMPLES = None     # Use all validation data (~1,000 conversations)
BPE_TRAIN_SAMPLES = 5000   # Sample for BPE training

# Conversation format config
INCLUDE_PERSONAS = False    # Set to True to include persona descriptions in training
TURN_MARKER_STYLE = "token"  # "token" or "text" - how to mark conversation turns


def get_pair_stats(tokens):
    """Get statistics of adjacent token pairs."""
    stats = defaultdict(int)
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        stats[pair] += 1
    return stats


def merge_tokens(tokens, pair, new_token):
    """Merge all occurrences of a token pair into a new token."""
    merged = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            merged.append(new_token)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def train_bpe_optimized(text, target_vocab_size, reserved_tokens=4):
    """
    Optimized BPE training with incremental pair statistics updates.
    Reserved tokens are slots for special tokens (U1, U2, END, PERSONA).
    """
    print(f"Training BPE on {len(text):,} characters...")
    tokens = list(text.replace(" ", SP_MARKER))
    vocab = sorted(set(tokens))
    vocab_set = set(vocab)
    merges = []

    # Adjust target to account for reserved special tokens
    adjusted_target = target_vocab_size - reserved_tokens

    if adjusted_target <= len(vocab):
        return vocab, merges

    print(f"Initial vocab size: {len(vocab)}")
    print(f"Target vocab size: {adjusted_target} (+ {reserved_tokens} special tokens)")

    # Initial pair statistics
    print("Computing initial pair statistics...")
    start_time = time.time()
    stats = get_pair_stats(tokens)
    print(f"  Done in {time.time() - start_time:.2f}s")

    iteration = 0
    while len(vocab) < adjusted_target:
        iteration += 1

        if not stats:
            print("No more pairs to merge")
            break

        # Find most frequent pair
        best_pair = max(stats, key=stats.get)
        best_count = stats[best_pair]
        new_token = "".join(best_pair)

        if new_token in vocab_set:
            print(f"Token '{new_token}' already exists, stopping")
            break

        # Progress reporting
        if len(vocab) % 50 == 0 or iteration <= 5:
            elapsed = time.time() - start_time
            rate = iteration / elapsed if elapsed > 0 else 0
            remaining = (adjusted_target - len(vocab)) / rate if rate > 0 else 0
            print(f"  Vocab: {len(vocab)}/{adjusted_target} | "
                  f"Merging '{best_pair[0]}{best_pair[1]}' ({best_count}x) | "
                  f"Speed: {rate:.1f} merges/s | ETA: {remaining:.0f}s")

        # Merge tokens efficiently
        new_tokens = []
        i = 0
        pairs_to_remove = defaultdict(int)
        pairs_to_add = defaultdict(int)

        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                if i > 0:
                    old_left_pair = (tokens[i - 1], tokens[i])
                    pairs_to_remove[old_left_pair] += 1
                if i + 2 < len(tokens):
                    old_right_pair = (tokens[i + 1], tokens[i + 2])
                    pairs_to_remove[old_right_pair] += 1

                new_tokens.append(new_token)

                if len(new_tokens) > 1:
                    new_left_pair = (new_tokens[-2], new_token)
                    pairs_to_add[new_left_pair] += 1
                if i + 2 < len(tokens):
                    new_right_pair = (new_token, tokens[i + 2])
                    pairs_to_add[new_right_pair] += 1

                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        # Update statistics incrementally
        del stats[best_pair]
        for pair, count in pairs_to_remove.items():
            stats[pair] = max(0, stats[pair] - count)
            if stats[pair] == 0:
                del stats[pair]
        for pair, count in pairs_to_add.items():
            stats[pair] += count

        tokens = new_tokens
        merges.append(best_pair)
        vocab.append(new_token)
        vocab_set.add(new_token)

    total_time = time.time() - start_time
    print(f"\n✓ BPE training completed in {total_time:.1f}s")
    return vocab, merges


def train_bpe(text, target_vocab_size):
    """Wrapper to use optimized BPE training."""
    return train_bpe_optimized(text, target_vocab_size, reserved_tokens=len(SPECIAL_TOKENS))


def encode(text, merges, stoi, show_progress=False):
    """Encode text with BPE merges (single text)."""
    tokens = list(text.replace(" ", SP_MARKER))

    for pair in merges:
        merged_token = "".join(pair)
        tokens = merge_tokens(tokens, pair, merged_token)

    # Filter unknown tokens
    encoded = [stoi[t] for t in tokens if t in stoi]
    return encoded


def encode_chunked(texts, merges, stoi, chunk_size=500, show_progress=True):
    """
    Encode multiple texts in chunks for better performance.
    Smaller chunk_size for conversations (they're longer than stories).
    """
    print(f"  Encoding {len(texts):,} conversations in chunks of {chunk_size:,}...")

    all_ids = []
    num_chunks = (len(texts) + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(texts))
        chunk_texts = texts[start_idx:end_idx]

        # Combine chunk into single string
        chunk_text = "\n".join(chunk_texts)

        # Encode the chunk
        chunk_ids = encode(chunk_text, merges, stoi, show_progress=False)
        all_ids.extend(chunk_ids)

        if show_progress:
            progress = (chunk_idx + 1) / num_chunks * 100
            print(f"    Chunk {chunk_idx + 1}/{num_chunks} ({progress:.1f}%) - "
                  f"{len(chunk_ids):,} tokens - Total: {len(all_ids):,} tokens")

    return all_ids


def decode(ids, itos):
    text = "".join(itos[i] for i in ids)
    return text.replace(SP_MARKER, " ")


def clean_text(text):
    """
    Clean text to ASCII-only (remove special Unicode characters).
    More permissive than TinyStories to preserve conversation markers.
    """
    allowed_chars = set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789'
        ' .,!?;:\'"()-\n\t'
    )

    cleaned = []
    for char in text:
        if char in allowed_chars:
            cleaned.append(char)

    return ''.join(cleaned)


def format_conversation(user1_persona, user2_persona, conversation, include_personas=False):
    """
    Format a PersonaChat conversation for training.

    Format: <U1>Hello!<U2>Hi there!<U1>How are you?<U2>Good!<END>
    """
    formatted = ""

    # Optionally include personas at the start
    if include_personas:
        formatted += f"<PERSONA>{clean_text(user1_persona)}<PERSONA>{clean_text(user2_persona)}"

    # Parse conversation and add turn markers
    # Conversation format in dataset: "User 1: text\nUser 2: text\n..."
    lines = conversation.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('User 1:') or line.startswith('user 1:'):
            text = line.split(':', 1)[1].strip()
            formatted += f"<U1>{clean_text(text)}"
        elif line.startswith('User 2:') or line.startswith('user 2:'):
            text = line.split(':', 1)[1].strip()
            formatted += f"<U2>{clean_text(text)}"
        else:
            # Handle other formats or continue previous turn
            formatted += clean_text(line)

    formatted += "<END>"
    return formatted


# Download and prepare PersonaChat dataset
print("="*60)
print("PersonaChat Dataset Preparation")
print("="*60)
print("Loading Synthetic-Persona-Chat dataset from HuggingFace...")
print("This may take a few minutes on first run (dataset will be cached)")
print()

try:
    from datasets import load_dataset

    # Load Synthetic-Persona-Chat dataset
    dataset = load_dataset("google/Synthetic-Persona-Chat")

    print(f"✓ Dataset loaded successfully!")
    print(f"  Train samples: {len(dataset['train']):,}")
    print(f"  Validation samples: {len(dataset['validation']):,}")
    print(f"  Test samples: {len(dataset['test']):,}")
    print()

    # Select samples
    if MAX_TRAIN_SAMPLES and len(dataset['train']) > MAX_TRAIN_SAMPLES:
        print(f"Limiting train samples to {MAX_TRAIN_SAMPLES:,}")
        train_dataset = dataset['train'].select(range(MAX_TRAIN_SAMPLES))
    else:
        train_dataset = dataset['train']

    if MAX_VAL_SAMPLES and len(dataset['validation']) > MAX_VAL_SAMPLES:
        print(f"Limiting validation samples to {MAX_VAL_SAMPLES:,}")
        val_dataset = dataset['validation'].select(range(MAX_VAL_SAMPLES))
    else:
        val_dataset = dataset['validation']

    # Format conversations
    print("Formatting conversations...")
    train_texts = []
    for example in train_dataset:
        formatted = format_conversation(
            example['user 1 personas'],
            example['user 2 personas'],
            example['Best Generated Conversation'],
            include_personas=INCLUDE_PERSONAS
        )
        train_texts.append(formatted)

    val_texts = []
    for example in val_dataset:
        formatted = format_conversation(
            example['user 1 personas'],
            example['user 2 personas'],
            example['Best Generated Conversation'],
            include_personas=INCLUDE_PERSONAS
        )
        val_texts.append(formatted)

    # For BPE training, concatenate conversations
    train_data = "\n".join(train_texts)
    val_data = "\n".join(val_texts)

except ImportError:
    print("="*60)
    print("ERROR: datasets library not installed!")
    print("="*60)
    print("Install with:")
    print("  pip install datasets")
    print("  or: uv pip install datasets")
    print()
    exit(1)
except Exception as e:
    print(f"ERROR: Failed to load dataset: {e}")
    exit(1)

print()
print(f"Dataset statistics:")
print(f"  Train: {len(train_texts):,} conversations, {len(train_data):,} characters")
print(f"  Val:   {len(val_texts):,} conversations, {len(val_data):,} characters")
print()

# Use smaller sample for BPE training (faster)
print(f"Preparing BPE training sample ({BPE_TRAIN_SAMPLES:,} conversations)...")
if BPE_TRAIN_SAMPLES and BPE_TRAIN_SAMPLES < len(train_texts):
    bpe_train_texts = train_texts[:BPE_TRAIN_SAMPLES]
    bpe_train_data = "\n".join(bpe_train_texts)
    print(f"  BPE training sample: {len(bpe_train_data):,} characters")
else:
    bpe_train_data = train_data
    print(f"  Using full training data for BPE")

print()

# Train BPE tokenizer
print("Training BPE tokenizer...")
vocab, merges = train_bpe(bpe_train_data, TARGET_VOCAB_SIZE)

# Add special tokens to vocabulary (at the beginning)
special_token_names = list(SPECIAL_TOKENS.keys())
full_vocab = special_token_names + vocab

# Create mappings
itos = {i: t for i, t in enumerate(full_vocab)}
stoi = {t: i for i, t in itos.items()}

print()
print(f"✓ BPE training complete!")
print(f"  Vocabulary size: {len(full_vocab):,} (including {len(SPECIAL_TOKENS)} special tokens)")
print(f"  Special tokens: {special_token_names}")
print(f"  Number of merges: {len(merges):,}")
print()

# Encode train/val using CHUNKED encoding
print("="*60)
print("Encoding train/val splits with CHUNKED processing...")
print("="*60)

print("\nEncoding training data...")
start_time = time.time()
train_ids = encode_chunked(train_texts, merges, stoi, chunk_size=500)
train_time = time.time() - start_time
print(f"  ✓ Train encoding done in {train_time:.1f}s ({len(train_ids):,} tokens)")

print("\nEncoding validation data...")
start_time = time.time()
val_ids = encode_chunked(val_texts, merges, stoi, chunk_size=500)
val_time = time.time() - start_time
print(f"  ✓ Val encoding done in {val_time:.1f}s ({len(val_ids):,} tokens)")

print()
print("="*60)
print(f"✓ Encoding complete!")
print(f"  Train: {len(train_ids):,} tokens")
print(f"  Val:   {len(val_ids):,} tokens")
print(f"  Total time: {train_time + val_time:.1f}s")
print("="*60)
print()

# Export to bin files
output_dir = os.path.dirname(__file__)
if not output_dir:
    output_dir = "."

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(output_dir, "train.bin"))
val_ids.tofile(os.path.join(output_dir, "val.bin"))

# Save the meta information
meta = {
    "vocab_size": len(full_vocab),
    "itos": itos,
    "stoi": stoi,
    "merges": merges,
    "level": "bpe",
    "bpe_type": "tiny_sp_conversation",
    "sp_marker": SP_MARKER,
    "special_tokens": SPECIAL_TOKENS,
    "dataset": "Synthetic-Persona-Chat",
    "source": "https://huggingface.co/datasets/google/Synthetic-Persona-Chat",
    "max_train_samples": MAX_TRAIN_SAMPLES,
    "max_val_samples": MAX_VAL_SAMPLES,
    "include_personas": INCLUDE_PERSONAS,
}
with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("="*60)
print("✓ Dataset preparation complete!")
print("="*60)
print(f"Output directory: {os.path.abspath(output_dir)}")
print()
print("Files created:")
print(f"  ✓ train.bin  ({len(train_ids):,} tokens)")
print(f"  ✓ val.bin    ({len(val_ids):,} tokens)")
print(f"  ✓ meta.pkl   (vocab + tokenizer)")
print()
print("Conversation format:")
print(f"  <U1>Hello!<U2>Hi!<U1>How are you?<U2>Good!<END>")
print()
print("Next steps:")
print(f"  cd {os.path.abspath(os.path.join(output_dir, '../..'))} ")
print(f"  python train_tf.py tf_config/train_pico_personachat_tiny_bpe.py")
print("="*60)
