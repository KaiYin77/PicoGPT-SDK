#!/usr/bin/env python3
"""
Parse training log and plot validation loss curve.

This script reads the cleaned training log (containing only checkpoint saves)
and generates a loss curve graph.

Usage:
    python plot_training_loss.py <log_file> [output_image]

Examples:
    # Auto-save to loss_curve.png in same directory as log
    python plot_training_loss.py out-tf-sub-pico-tinystories-tiny-bpe/trainval.log

    # Specify custom output filename
    python plot_training_loss.py trainval.log my_loss_curve.png
"""

import sys
import re
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def parse_log_file(log_file):
    """
    Parse cleaned log file and extract validation losses.

    Expected format:
    └─ Saved checkpoint to out-tf-sub-pico-tinystories-tiny-bpe (val_loss: 2.1341)
    """
    val_losses = []

    # Pattern to extract val_loss
    pattern = re.compile(r'val_loss:\s*([\d.]+)')

    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            match = pattern.search(line)
            if match:
                val_loss = float(match.group(1))
                val_losses.append(val_loss)

    return val_losses


def plot_loss_curve(val_losses, output_file, title=None):
    """
    Plot validation loss curve and save to file.
    """
    if not val_losses:
        print("ERROR: No validation losses found in log file!")
        return False

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot validation loss
    checkpoints = list(range(1, len(val_losses) + 1))
    plt.plot(checkpoints, val_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Validation Loss')

    # Find best (minimum) loss
    best_idx = val_losses.index(min(val_losses))
    best_loss = val_losses[best_idx]

    # Mark best checkpoint
    plt.plot(best_idx + 1, best_loss, 'r*', markersize=15, label=f'Best: {best_loss:.4f}')

    # Labels and title
    plt.xlabel('Checkpoint Number', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)

    if title is None:
        title = 'Training Progress - Validation Loss'
    plt.title(title, fontsize=14, fontweight='bold')

    # Grid
    plt.grid(True, alpha=0.3, linestyle='--')

    # Legend
    plt.legend(loc='best', fontsize=10)

    # Add statistics text box
    stats_text = f'Total Checkpoints: {len(val_losses)}\n'
    stats_text += f'Best Loss: {best_loss:.4f} (Checkpoint {best_idx + 1})\n'
    stats_text += f'Final Loss: {val_losses[-1]:.4f}\n'
    stats_text += f'Improvement: {val_losses[0] - val_losses[-1]:.4f}'

    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Loss curve saved to: {output_file}")

    # Print statistics
    print()
    print("="*60)
    print("Training Statistics")
    print("="*60)
    print(f"Total checkpoints: {len(val_losses)}")
    print(f"Initial loss:      {val_losses[0]:.4f}")
    print(f"Final loss:        {val_losses[-1]:.4f}")
    print(f"Best loss:         {best_loss:.4f} (checkpoint {best_idx + 1})")
    print(f"Total improvement: {val_losses[0] - val_losses[-1]:.4f}")
    print(f"Reduction:         {(1 - val_losses[-1] / val_losses[0]) * 100:.2f}%")
    print("="*60)

    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    log_file = sys.argv[1]

    if not os.path.exists(log_file):
        print(f"ERROR: File not found: {log_file}")
        sys.exit(1)

    # Determine output file
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        # Auto-save to loss_curve.png in same directory as log
        log_dir = os.path.dirname(log_file) or '.'
        output_file = os.path.join(log_dir, 'loss_curve.png')

    # Extract model name for title
    log_dir = os.path.dirname(log_file)
    model_name = os.path.basename(log_dir) if log_dir else 'Training'
    title = f'Training Progress - {model_name}'

    # Parse log file
    print(f"Reading: {log_file}")
    val_losses = parse_log_file(log_file)

    if not val_losses:
        print("ERROR: No validation losses found!")
        print("Expected format: └─ Saved checkpoint to ... (val_loss: 2.1341)")
        sys.exit(1)

    print(f"Found {len(val_losses)} checkpoints")

    # Plot and save
    success = plot_loss_curve(val_losses, output_file, title)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
