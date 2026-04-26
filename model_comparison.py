"""
Model-to-Model Comparison with Plotted Images
Shows honest comparison: Original Single-task vs Extended Multi-task
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"c:\Users\zahid\Downloads\ANN Project part2")
RESULTS_DIR = BASE_DIR / "comparison_results"
OUTPUT_DIR = RESULTS_DIR


def load_result(path):
    with open(path) as f:
        return json.load(f)


def plot_classification_comparison():
    """Figure 1: Classification Accuracy & F1 Comparison (Bar Chart)"""
    orig_msoud = load_result(RESULTS_DIR / "original_baseline_Msoud" / "results.json")
    orig_neuro = load_result(RESULTS_DIR / "original_baseline_NeuroMRI" / "results.json")
    ext_msoud = load_result(RESULTS_DIR / "extended_multitask_Msoud" / "results.json")
    ext_neuro = load_result(RESULTS_DIR / "extended_multitask_NeuroMRI" / "results.json")
    ext_epic = load_result(RESULTS_DIR / "extended_multitask_Epic" / "results.json")

    datasets = ['Msoud', 'NeuroMRI', 'Epic']
    
    # Accuracy values
    orig_acc = [orig_msoud['accuracy'], orig_neuro['accuracy'], 0]
    ext_acc = [ext_msoud['classification']['accuracy'], 
               ext_neuro['classification']['accuracy'],
               ext_epic['classification']['accuracy']]
    
    # F1 values
    orig_f1 = [orig_msoud['f1'], orig_neuro['f1'], 0]
    ext_f1 = [ext_msoud['classification']['f1'],
              ext_neuro['classification']['f1'],
              ext_epic['classification']['f1']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(datasets))
    width = 0.35

    # Accuracy plot
    bars1 = axes[0].bar(x - width/2, orig_acc, width, label='Original (Single-task)', color='#1976D2', edgecolor='black')
    bars2 = axes[0].bar(x + width/2, ext_acc, width, label='Extended (Multi-task)', color='#E65100', edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=13)
    axes[0].set_title('Classification Accuracy Comparison', fontsize=15, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].set_ylim(0, 1.0)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            axes[0].annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    for bar in bars2:
        h = bar.get_height()
        axes[0].annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    # F1 plot
    bars3 = axes[1].bar(x - width/2, orig_f1, width, label='Original (Single-task)', color='#1976D2', edgecolor='black')
    bars4 = axes[1].bar(x + width/2, ext_f1, width, label='Extended (Multi-task)', color='#E65100', edgecolor='black')
    axes[1].set_ylabel('F1 Score', fontsize=13)
    axes[1].set_title('Classification F1 Score Comparison', fontsize=15, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].set_ylim(0, 1.0)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(axis='y', alpha=0.3)

    for bar in bars3:
        h = bar.get_height()
        if h > 0:
            axes[1].annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    for bar in bars4:
        h = bar.get_height()
        axes[1].annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    plt.suptitle('ORIGINAL vs EXTENDED: Classification Performance\n(HONEST COMPARISON - Msoud Shows Downfall)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison_classification.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison_classification.png")


def plot_performance_change():
    """Figure 2: Performance Change Percentage with Color Coding"""
    orig_msoud = load_result(RESULTS_DIR / "original_baseline_Msoud" / "results.json")
    orig_neuro = load_result(RESULTS_DIR / "original_baseline_NeuroMRI" / "results.json")
    ext_msoud = load_result(RESULTS_DIR / "extended_multitask_Msoud" / "results.json")
    ext_neuro = load_result(RESULTS_DIR / "extended_multitask_NeuroMRI" / "results.json")

    # Calculate changes for datasets where both exist
    msoud_f1_change = ((ext_msoud['classification']['f1'] - orig_msoud['f1']) / orig_msoud['f1']) * 100
    neuro_f1_change = ((ext_neuro['classification']['f1'] - orig_neuro['f1']) / orig_neuro['f1']) * 100

    datasets = ['Msoud', 'NeuroMRI']
    changes = [msoud_f1_change, neuro_f1_change]
    colors = ['#D32F2F' if c < 0 else '#388E3C' for c in changes]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(datasets, changes, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel('F1 Score Change (%)', fontsize=13)
    ax.set_title('Extended vs Original: F1 Score Change\nRED = Downfall  |  GREEN = Improvement', 
                 fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        h = bar.get_height()
        offset = 3 if h >= 0 else -15
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, offset), textcoords="offset points", ha='center', 
                   fontsize=14, fontweight='bold')

    # Add explanation text
    ax.text(0.5, -25, 'Msoud: -33.7% downfall (multi-task trade-off)\nNeuroMRI: +18.6% improvement (auxiliary task helps small dataset)',
           ha='center', fontsize=11, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.ylim(-50, 50)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison_f1_change.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison_f1_change.png")


def plot_all_metrics_radar():
    """Figure 3: Radar Chart comparing Original vs Extended on shared datasets"""
    orig_msoud = load_result(RESULTS_DIR / "original_baseline_Msoud" / "results.json")
    orig_neuro = load_result(RESULTS_DIR / "original_baseline_NeuroMRI" / "results.json")
    ext_msoud = load_result(RESULTS_DIR / "extended_multitask_Msoud" / "results.json")
    ext_neuro = load_result(RESULTS_DIR / "extended_multitask_NeuroMRI" / "results.json")

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    # Msoud values
    orig_msoud_vals = [orig_msoud['accuracy'], orig_msoud['precision'], 
                       orig_msoud['recall'], orig_msoud['f1']]
    ext_msoud_vals = [ext_msoud['classification']['accuracy'], ext_msoud['classification']['precision'],
                      ext_msoud['classification']['recall'], ext_msoud['classification']['f1']]
    
    # NeuroMRI values
    orig_neuro_vals = [orig_neuro['accuracy'], orig_neuro['precision'],
                       orig_neuro['recall'], orig_neuro['f1']]
    ext_neuro_vals = [ext_neuro['classification']['accuracy'], ext_neuro['classification']['precision'],
                      ext_neuro['classification']['recall'], ext_neuro['classification']['f1']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for ax, orig_vals, ext_vals, title in zip(
        axes, 
        [orig_msoud_vals, orig_neuro_vals], 
        [ext_msoud_vals, ext_neuro_vals],
        ['Msoud Dataset', 'NeuroMRI Dataset']
    ):
        orig_vals += orig_vals[:1]
        ext_vals += ext_vals[:1]
        
        ax.plot(angles, orig_vals, 'o-', linewidth=2, label='Original (Single-task)', color='#1976D2')
        ax.fill(angles, orig_vals, alpha=0.15, color='#1976D2')
        ax.plot(angles, ext_vals, 'o-', linewidth=2, label='Extended (Multi-task)', color='#E65100')
        ax.fill(angles, ext_vals, alpha=0.15, color='#E65100')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)

    plt.suptitle('Original vs Extended: All Metrics Radar Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison_radar.png")


def plot_grading_vs_classification():
    """Figure 4: Extended Model - Classification vs Grading Performance"""
    ext_msoud = load_result(RESULTS_DIR / "extended_multitask_Msoud" / "results.json")
    ext_neuro = load_result(RESULTS_DIR / "extended_multitask_NeuroMRI" / "results.json")
    ext_epic = load_result(RESULTS_DIR / "extended_multitask_Epic" / "results.json")

    datasets = ['Msoud', 'NeuroMRI', 'Epic']
    cls_acc = [ext_msoud['classification']['accuracy'],
               ext_neuro['classification']['accuracy'],
               ext_epic['classification']['accuracy']]
    grd_acc = [ext_msoud['grading']['accuracy'],
               ext_neuro['grading']['accuracy'],
               ext_epic['grading']['accuracy']]
    cls_f1 = [ext_msoud['classification']['f1'],
              ext_neuro['classification']['f1'],
              ext_epic['classification']['f1']]
    grd_f1 = [ext_msoud['grading']['f1'],
              ext_neuro['grading']['f1'],
              ext_epic['grading']['f1']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(datasets))
    width = 0.35

    # Accuracy
    bars1 = axes[0].bar(x - width/2, cls_acc, width, label='Classification', color='#1565C0', edgecolor='black')
    bars2 = axes[0].bar(x + width/2, grd_acc, width, label='Grading', color='#C62828', edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=13)
    axes[0].set_title('Extended Model: Classification vs Grading Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis='y', alpha=0.3)
    for bar in bars1 + bars2:
        h = bar.get_height()
        axes[0].annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    # F1
    bars3 = axes[1].bar(x - width/2, cls_f1, width, label='Classification', color='#1565C0', edgecolor='black')
    bars4 = axes[1].bar(x + width/2, grd_f1, width, label='Grading', color='#C62828', edgecolor='black')
    axes[1].set_ylabel('F1 Score', fontsize=13)
    axes[1].set_title('Extended Model: Classification vs Grading F1', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis='y', alpha=0.3)
    for bar in bars3 + bars4:
        h = bar.get_height()
        axes[1].annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    plt.suptitle('Extended Multi-Task Model: Both Task Performances\n(Grading is a NOVEL capability not present in Original)', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison_tasks.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison_tasks.png")


def generate_comparison_table_image():
    """Figure 5: Detailed Comparison Table as Image"""
    orig_msoud = load_result(RESULTS_DIR / "original_baseline_Msoud" / "results.json")
    orig_neuro = load_result(RESULTS_DIR / "original_baseline_NeuroMRI" / "results.json")
    ext_msoud = load_result(RESULTS_DIR / "extended_multitask_Msoud" / "results.json")
    ext_neuro = load_result(RESULTS_DIR / "extended_multitask_NeuroMRI" / "results.json")
    ext_epic = load_result(RESULTS_DIR / "extended_multitask_Epic" / "results.json")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    table_data = [
        ['Dataset', 'Work', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Grading Acc', 'Grading F1'],
        ['Msoud', 'Original', 'Single-task', f"{orig_msoud['accuracy']:.4f}", 
         f"{orig_msoud['precision']:.4f}", f"{orig_msoud['recall']:.4f}", 
         f"{orig_msoud['f1']:.4f}", 'N/A', 'N/A'],
        ['Msoud', 'Extended', 'Multi-task', f"{ext_msoud['classification']['accuracy']:.4f}",
         f"{ext_msoud['classification']['precision']:.4f}", f"{ext_msoud['classification']['recall']:.4f}",
         f"{ext_msoud['classification']['f1']:.4f}", f"{ext_msoud['grading']['accuracy']:.4f}",
         f"{ext_msoud['grading']['f1']:.4f}"],
        ['NeuroMRI', 'Original', 'Single-task', f"{orig_neuro['accuracy']:.4f}",
         f"{orig_neuro['precision']:.4f}", f"{orig_neuro['recall']:.4f}",
         f"{orig_neuro['f1']:.4f}", 'N/A', 'N/A'],
        ['NeuroMRI', 'Extended', 'Multi-task', f"{ext_neuro['classification']['accuracy']:.4f}",
         f"{ext_neuro['classification']['precision']:.4f}", f"{ext_neuro['classification']['recall']:.4f}",
         f"{ext_neuro['classification']['f1']:.4f}", f"{ext_neuro['grading']['accuracy']:.4f}",
         f"{ext_neuro['grading']['f1']:.4f}"],
        ['Epic', 'Extended', 'Multi-task', f"{ext_epic['classification']['accuracy']:.4f}",
         f"{ext_epic['classification']['precision']:.4f}", f"{ext_epic['classification']['recall']:.4f}",
         f"{ext_epic['classification']['f1']:.4f}", f"{ext_epic['grading']['accuracy']:.4f}",
         f"{ext_epic['grading']['f1']:.4f}"],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.1, 0.1, 0.12, 0.1, 0.1, 0.1, 0.1, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(9):
        cell = table[(0, i)]
        cell.set_facecolor('#1976D2')
        cell.set_text_props(weight='bold', color='white')

    # Style rows
    colors = ['#E3F2FD', '#FFF3E0']
    for i in range(1, 6):
        for j in range(9):
            table[(i, j)].set_facecolor(colors[i % 2])

    # Highlight downfall row (Msoud Extended)
    for j in range(9):
        table[(2, j)].set_facecolor('#FFCDD2')

    # Highlight improvement row (NeuroMRI Extended)
    for j in range(9):
        table[(4, j)].set_facecolor('#C8E6C9')

    ax.set_title('COMPLETE MODEL-TO-MODEL COMPARISON TABLE\nRed = Downfall  |  Green = Improvement  |  Blue = Original Baseline', 
                 fontsize=15, fontweight='bold', pad=20)

    # Add note
    fig.text(0.5, 0.02, 
             'NOTE: Results from 5-epoch quick demonstration. Full 40-epoch training required for final comparison.\n'
             'KEY FINDING: Msoud classification drops -33.7% F1 (multi-task trade-off). NeuroMRI improves +18.6% F1.',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison_table.png")


if __name__ == "__main__":
    plot_classification_comparison()
    plot_performance_change()
    plot_all_metrics_radar()
    plot_grading_vs_classification()
    generate_comparison_table_image()
    print("\nAll model comparison images generated successfully!")
