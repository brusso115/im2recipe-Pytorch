import json
import os

import matplotlib.pyplot as plt
import pandas as pd

def make_curve(learning_curves_file: str, field: str, output_dir='plots/', title=None):
    """Make a learning curve from a learning curve file.
    
    `field` should be the loss field from the json (e.g. "cos_loss").
    """
    df = pd.read_json(learning_curves_file)
    plt.plot(df['epoch'], df[field], '-o')
    plt.xlabel('epoch')
    plt.ylabel(field)
    # Add a little space at the bottom so the plot doesn't look so cramped.
    buffer = 0.05 * (df[field].max())
    plt.ylim(0 - buffer, df[field].max() + buffer)

    # plt.ylim(bottom=0)
    plt.grid()
    if title is None:
        title = field
    plt.title(title)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, field + '.png'))
    plt.close()


def make_curves(learning_curves_file: str, output_dir='plots/', title=None):
    """Put all the learning curves in one figure as subplots."""
    df = pd.read_json(learning_curves_file)
    fields = [col for col in df.columns if col != 'epoch']
    n_fields = len(fields)
    fig, axes = plt.subplots(n_fields // 3 + 1, 3, figsize=(15, 5 * (n_fields // 3 + 1)))
    for i, field in enumerate(fields):
        row = i // 3
        col = i % 3
        axes[row, col].plot(df['epoch'], df[field], '-o')
        axes[row, col].set_xlabel('epoch')
        axes[row, col].set_ylabel(field)
        buffer = 0.05 * (df[field].max())
        axes[row, col].set_ylim(0 - buffer, df[field].max() + buffer)
        axes[row, col].grid()
        axes[row, col].set_title(field)
    plt.suptitle(title)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'all.png'))
    plt.close()

def make_train_val_curve(learning_curves_file: str, output_dir='plots/', title=None):
    """Make a learning curve from a learning curve file."""
    df = pd.read_json(learning_curves_file)
    
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(df['epoch'], df['combined_loss'], '-o', label='train', color='tab:blue', alpha=0.7)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Combined Train Loss', color='tab:blue')


    ax2 = ax1.twinx()
    ax2.plot(df['epoch'], df['val_median_recall'], '-o', label='val', color='tab:red', alpha=0.7)
    ax2.set_ylabel('Val Median Recall', color='tab:red')

    # Add a little space at the bottom so the plot doesn't look so cramped.
    buffer1 = 0.05 * (df['combined_loss'].max())
    buffer2 = 0.05 * (df['val_median_recall'].max())
    ax1.set_ylim(0 - buffer1, df['combined_loss'].max() + buffer1)
    ax2.set_ylim(0 - buffer2, df['val_median_recall'].max() + buffer2)

    plt.grid()
    if title is None:
        title = 'train_val_loss'
    plt.title(title)
    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower left')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'train_val_loss.png'))
    plt.close()


if __name__ == '__main__':
    filepath = 'learning_curves/curves_12_11_21_36_00.json'
    make_curve(filepath, 'cos_loss', output_dir='plots/combined_improvements_30')
    make_curves(filepath, output_dir='plots/combined_improvements_30', title='vision + text transformer 10 class data')
    
