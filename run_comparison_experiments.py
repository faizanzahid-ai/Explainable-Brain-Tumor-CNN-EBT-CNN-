"""
Run original paper baseline (Msoud + NeuroMRI, single-task CNN) vs
Extended work (Msoud + Epic + NeuroMRI, multi-task CNN)
and generate comparison tables, figures, and graphs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import tensorflow as tf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

from model import create_model, create_multitask_model
from preprocessing import BrainMRIPreprocessor, load_and_preprocess_dataset
from train import assign_grade_labels

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORIGINAL_DATASETS = ['Msoud', 'NeuroMRI']
EXTENDED_DATASETS = ['Msoud', 'Epic', 'NeuroMRI']

DATASETS = {
    'Msoud': {
        'train_dir': r"c:\Users\zahid\Downloads\ANN Project part2\Msoud\Training",
        'test_dir':  r"c:\Users\zahid\Downloads\ANN Project part2\Msoud\Testing",
    },
    'Epic': {
        'train_dir': r"c:\Users\zahid\Downloads\ANN Project part2\Epic and CSCR hospital Dataset\Epic and CSCR hospital Dataset\Train",
        'test_dir':  r"c:\Users\zahid\Downloads\ANN Project part2\Epic and CSCR hospital Dataset\Epic and CSCR hospital Dataset\Test",
    },
    'NeuroMRI': {
        'train_dir': r"c:\Users\zahid\Downloads\ANN Project part2\NeuroMRI\Training",
        'test_dir':  r"c:\Users\zahid\Downloads\ANN Project part2\NeuroMRI\Testing",
    },
}

OUTPUT_DIR = Path(r"c:\Users\zahid\Downloads\ANN Project part2\comparison_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ['Meningioma', 'No Tumor', 'Glioma', 'Pituitary']

BATCH_SIZE = 40
LR = 0.001
EPOCHS = 40  # As used in the research paper

WORK_COLORS = {'Original Paper': '#1976D2', 'Extended Work': '#E65100'}

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_dataset(ds_name):
    ds = DATASETS[ds_name]
    preprocessor = BrainMRIPreprocessor(target_size=(224, 224))
    train_dir = Path(ds['train_dir'])
    test_dir = Path(ds['test_dir'])

    if not train_dir.exists() or not test_dir.exists():
        print(f"  ERROR: Dataset directories not found for {ds_name}")
        return None, None, None, None

    print(f"  Loading training data from {train_dir}...")
    X_train, y_train, _ = load_and_preprocess_dataset(train_dir, preprocessor, classes=CLASSES)
    print(f"  Loading test data from {test_dir}...")
    X_test, y_test, _ = load_and_preprocess_dataset(test_dir, preprocessor, classes=CLASSES)

    X_train = X_train.astype('float16') / 255.0
    X_test = X_test.astype('float16') / 255.0

    print(f"  Train: {len(X_train)} images, Test: {len(X_test)} images")
    return X_train, y_train, X_test, y_test

# ---------------------------------------------------------------------------
# Experiment Runners
# ---------------------------------------------------------------------------

def run_original_single_task(ds_name):
    print(f"\n{'='*70}")
    print(f"  ORIGINAL PAPER: Single-task Baseline on {ds_name}")
    print(f"{'='*70}")

    X_train, y_train, X_test, y_test = load_dataset(ds_name)
    if X_train is None:
        return None

    model = create_model(input_shape=(224, 224, 3), num_classes=4, learning_rate=LR)
    print(f"  Training for {EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LR})...")

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    accuracy = np.mean(y_pred == y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    result = {
        'work_type': 'Original Paper',
        'model_type': 'Single-task Classification',
        'dataset': ds_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'epochs': EPOCHS
    }

    print(f"\n  Results: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")

    model_dir = OUTPUT_DIR / f"original_baseline_{ds_name}"
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir / 'model.keras')
    with open(model_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=4)

    del model
    tf.keras.backend.clear_session()
    return result


def run_extended_multitask(ds_name):
    print(f"\n{'='*70}")
    print(f"  EXTENDED WORK: Multi-task on {ds_name}")
    print(f"{'='*70}")

    X_train, y_train, X_test, y_test = load_dataset(ds_name)
    if X_train is None:
        return None

    y_grade_train = assign_grade_labels(y_train, CLASSES)
    y_grade_test = assign_grade_labels(y_test, CLASSES)

    model = create_multitask_model(input_shape=(224, 224, 3), num_classes=4, num_grades=4, learning_rate=LR)
    print(f"  Training for {EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LR})...")

    train_data = {'classification': y_train, 'grading': y_grade_train}

    history = model.fit(
        X_train, train_data,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    preds = model.predict(X_test, verbose=0)
    cls_pred = np.argmax(preds[0], axis=1)
    grade_pred = np.argmax(preds[1], axis=1)

    from sklearn.metrics import precision_score, recall_score, f1_score
    cls_accuracy = np.mean(cls_pred == y_test)
    cls_precision = precision_score(y_test, cls_pred, average='macro', zero_division=0)
    cls_recall = recall_score(y_test, cls_pred, average='macro', zero_division=0)
    cls_f1 = f1_score(y_test, cls_pred, average='macro', zero_division=0)

    grade_accuracy = np.mean(grade_pred == y_grade_test)
    grade_precision = precision_score(y_grade_test, grade_pred, average='macro', zero_division=0)
    grade_recall = recall_score(y_grade_test, grade_pred, average='macro', zero_division=0)
    grade_f1 = f1_score(y_grade_test, grade_pred, average='macro', zero_division=0)

    result = {
        'work_type': 'Extended Work',
        'model_type': 'Multi-task (Cls+Grading)',
        'dataset': ds_name,
        'classification': {
            'accuracy': float(cls_accuracy),
            'precision': float(cls_precision),
            'recall': float(cls_recall),
            'f1': float(cls_f1)
        },
        'grading': {
            'accuracy': float(grade_accuracy),
            'precision': float(grade_precision),
            'recall': float(grade_recall),
            'f1': float(grade_f1)
        },
        'epochs': EPOCHS
    }

    print(f"\n  Classification: Acc={cls_accuracy:.4f}, F1={cls_f1:.4f}")
    print(f"  Grading:        Acc={grade_accuracy:.4f}, F1={grade_f1:.4f}")

    model_dir = OUTPUT_DIR / f"extended_multitask_{ds_name}"
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir / 'model.keras')
    with open(model_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=4)

    del model
    tf.keras.backend.clear_session()
    return result

# ---------------------------------------------------------------------------
# Comparison Plotting
# ---------------------------------------------------------------------------

def load_all_results():
    results = {'original': {}, 'extended': {}}
    for ds in ORIGINAL_DATASETS:
        p = OUTPUT_DIR / f"original_baseline_{ds}" / "results.json"
        if p.exists():
            with open(p) as f:
                results['original'][ds] = json.load(f)
    for ds in EXTENDED_DATASETS:
        p = OUTPUT_DIR / f"extended_multitask_{ds}" / "results.json"
        if p.exists():
            with open(p) as f:
                results['extended'][ds] = json.load(f)
    return results


def create_dataframe(results):
    rows = []
    for ds, r in results['original'].items():
        rows.append({'Work': 'Original Paper', 'Dataset': ds, 'Accuracy': r['accuracy'],
                     'Precision': r['precision'], 'Recall': r['recall'], 'F1': r['f1'],
                     'Grading_Acc': None, 'Grading_F1': None})
    for ds, r in results['extended'].items():
        c = r['classification']
        g = r['grading']
        rows.append({'Work': 'Extended Work', 'Dataset': ds, 'Accuracy': c['accuracy'],
                     'Precision': c['precision'], 'Recall': c['recall'], 'F1': c['f1'],
                     'Grading_Acc': g['accuracy'], 'Grading_F1': g['f1']})
    return pd.DataFrame(rows)


def plot_table1(df):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis('off')
    data = []
    for _, row in df.iterrows():
        g = f"Acc={row['Grading_Acc']:.4f}\nF1={row['Grading_F1']:.4f}" if pd.notna(row['Grading_Acc']) else "N/A"
        data.append([row['Work'], row['Dataset'],
                     'Single-task CNN' if row['Work'] == 'Original Paper' else 'Multi-task CNN',
                     f"{row['Accuracy']:.4f}", f"{row['Precision']:.4f}",
                     f"{row['Recall']:.4f}", f"{row['F1']:.4f}", g])
    cols = ['Work', 'Dataset', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'WHO Grading']
    tab = ax.table(cellText=data, colLabels=cols, loc='center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1.1, 2.2)
    for i in range(len(data)):
        color = '#E3F2FD' if data[i][0] == 'Original Paper' else '#FFF3E0'
        for j in range(len(cols)):
            tab[(i+1, j)].set_facecolor(color)
    for j in range(len(cols)):
        tab[(0, j)].set_facecolor('#37474F')
        tab[(0, j)].set_text_props(color='white', fontweight='bold')
    ax.set_title('Table 1: Original Paper vs Extended Work Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'table1_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    df.to_csv(OUTPUT_DIR / 'table1_comparison.csv', index=False)
    print("  Saved table1_comparison")


def plot_fig1_accuracy(results):
    fig, ax = plt.subplots(figsize=(12, 7))
    datasets = ['Msoud', 'NeuroMRI', 'Epic']
    x = np.arange(len(datasets))
    width = 0.35
    orig_acc = [results['original'].get(ds, {}).get('accuracy', 0) for ds in datasets]
    ext_acc = [results['extended'].get(ds, {}).get('classification', {}).get('accuracy', 0) for ds in datasets]
    bars1 = ax.bar(x - width/2, orig_acc, width, label='Original Paper (Single-task)',
                   color=WORK_COLORS['Original Paper'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, ext_acc, width, label='Extended Work (Multi-task)',
                   color=WORK_COLORS['Extended Work'], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Figure 1: Classification Accuracy Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars1 + bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure1_accuracy_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved figure1_accuracy_comparison")


def plot_fig2_f1(results):
    fig, ax = plt.subplots(figsize=(12, 7))
    datasets = ['Msoud', 'NeuroMRI', 'Epic']
    x = np.arange(len(datasets))
    width = 0.35
    orig_f1 = [results['original'].get(ds, {}).get('f1', 0) for ds in datasets]
    ext_f1 = [results['extended'].get(ds, {}).get('classification', {}).get('f1', 0) for ds in datasets]
    bars1 = ax.bar(x - width/2, orig_f1, width, label='Original Paper (Single-task)',
                   color=WORK_COLORS['Original Paper'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, ext_f1, width, label='Extended Work (Multi-task)',
                   color=WORK_COLORS['Extended Work'], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('Figure 2: Classification F1 Score Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars1 + bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure2_f1_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved figure2_f1_comparison")


def plot_fig3_precision_recall(results):
    fig, ax = plt.subplots(figsize=(12, 7))
    datasets = ['Msoud', 'NeuroMRI', 'Epic']
    x = np.arange(len(datasets))
    width = 0.15
    orig_prec = [results['original'].get(ds, {}).get('precision', 0) for ds in datasets]
    orig_rec = [results['original'].get(ds, {}).get('recall', 0) for ds in datasets]
    ext_prec = [results['extended'].get(ds, {}).get('classification', {}).get('precision', 0) for ds in datasets]
    ext_rec = [results['extended'].get(ds, {}).get('classification', {}).get('recall', 0) for ds in datasets]
    ax.bar(x - 1.5*width, orig_prec, width, label='Original Precision', color='#1976D2', alpha=0.8)
    ax.bar(x - 0.5*width, orig_rec, width, label='Original Recall', color='#64B5F6', alpha=0.8)
    ax.bar(x + 0.5*width, ext_prec, width, label='Extended Precision', color='#E65100', alpha=0.8)
    ax.bar(x + 1.5*width, ext_rec, width, label='Extended Recall', color='#FFB74D', alpha=0.8)
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Figure 3: Precision and Recall Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(ncol=2)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure3_precision_recall.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved figure3_precision_recall")


def plot_fig4_all_metrics(results):
    fig, ax = plt.subplots(figsize=(14, 8))
    datasets = ['Msoud', 'NeuroMRI', 'Epic']
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(datasets))
    width = 0.1
    colors_orig = ['#0D47A1', '#1976D2', '#42A5F5', '#90CAF9']
    colors_ext = ['#BF360C', '#E65100', '#FF9800', '#FFCC80']
    for i, metric in enumerate(metrics_names):
        orig_vals = []
        ext_vals = []
        for ds in datasets:
            if metric == 'Accuracy':
                orig_vals.append(results['original'].get(ds, {}).get('accuracy', 0))
                ext_vals.append(results['extended'].get(ds, {}).get('classification', {}).get('accuracy', 0))
            else:
                orig_vals.append(results['original'].get(ds, {}).get(metric.lower(), 0))
                ext_vals.append(results['extended'].get(ds, {}).get('classification', {}).get(metric.lower(), 0))
        offset = (i - 1.5) * width * 2.2
        ax.bar(x + offset - width/2, orig_vals, width, color=colors_orig[i],
               label=f'Original {metric}', edgecolor='black', linewidth=0.3)
        ax.bar(x + offset + width/2, ext_vals, width, color=colors_ext[i],
               label=f'Extended {metric}', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Figure 4: All Metrics Comparison by Dataset', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.08))
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure4_all_metrics_grouped.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved figure4_all_metrics_grouped")


def plot_fig5_grading(results):
    fig, ax = plt.subplots(figsize=(12, 7))
    datasets = [ds for ds in EXTENDED_DATASETS if ds in results['extended']]
    x = np.arange(len(datasets))
    width = 0.35
    grading_acc = [results['extended'][ds]['grading']['accuracy'] for ds in datasets]
    grading_f1 = [results['extended'][ds]['grading']['f1'] for ds in datasets]
    bars1 = ax.bar(x - width/2, grading_acc, width, label='Accuracy', color='#2E7D32', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, grading_f1, width, label='F1 Score', color='#C62828', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Figure 5: Extended Work - WHO Tumor Grading Performance (Novel Capability)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars1 + bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure5_grading_performance.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved figure5_grading_performance")


def plot_fig6_radar(results):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    datasets_shared = [ds for ds in ORIGINAL_DATASETS if ds in results['original'] and ds in results['extended']]
    categories = ['Accuracy', 'Precision', 'Recall', 'F1']
    orig_scores = []
    ext_scores = []
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        vals = [results['original'][ds].get(metric, 0) for ds in datasets_shared]
        orig_scores.append(np.mean(vals) if vals else 0)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        vals = [results['extended'][ds]['classification'].get(metric, 0) for ds in datasets_shared]
        ext_scores.append(np.mean(vals) if vals else 0)
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    orig_scores += orig_scores[:1]
    ext_scores += ext_scores[:1]
    angles += angles[:1]
    ax.plot(angles, orig_scores, 'o-', linewidth=2.5, label='Original Paper', color=WORK_COLORS['Original Paper'])
    ax.fill(angles, orig_scores, alpha=0.2, color=WORK_COLORS['Original Paper'])
    ax.plot(angles, ext_scores, 's--', linewidth=2.5, label='Extended Work', color=WORK_COLORS['Extended Work'])
    ax.fill(angles, ext_scores, alpha=0.2, color=WORK_COLORS['Extended Work'])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title('Figure 6: Average Performance Radar Chart\n(Msoud + NeuroMRI)', fontsize=14, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure6_radar_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved figure6_radar_comparison")


def plot_fig7_performance_change(results):
    fig, ax = plt.subplots(figsize=(12, 7))
    datasets = [ds for ds in ORIGINAL_DATASETS if ds in results['original'] and ds in results['extended']]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(datasets))
    width = 0.2
    for i, metric in enumerate(metrics):
        changes = []
        for ds in datasets:
            if metric == 'Accuracy':
                orig = results['original'][ds].get('accuracy', 0)
                ext = results['extended'][ds]['classification'].get('accuracy', 0)
            else:
                orig = results['original'][ds].get(metric.lower(), 0)
                ext = results['extended'][ds]['classification'].get(metric.lower(), 0)
            change = ((ext - orig) / orig * 100) if orig > 0 else 0
            changes.append(change)
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, changes, width, label=metric)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + (1 if h >= 0 else -3),
                    f'{h:+.1f}%', ha='center', va='bottom' if h >= 0 else 'top',
                    fontsize=8, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Performance Change (%)', fontweight='bold')
    ax.set_title('Figure 7: Extended vs Original - Performance Change Percentage', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure7_performance_change.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved figure7_performance_change")


def plot_fig8_dataset_coverage(results):
    fig, ax = plt.subplots(figsize=(10, 7))
    categories = ['Original Paper', 'Extended Work']
    dataset_counts = [len(results['original']), len(results['extended'])]
    colors = [WORK_COLORS['Original Paper'], WORK_COLORS['Extended Work']]
    bars = ax.barh(categories, dataset_counts, color=colors, edgecolor='black', linewidth=0.5, height=0.5)
    ax.set_xlabel('Number of Datasets Evaluated', fontweight='bold')
    ax.set_title('Figure 8: Dataset Coverage Comparison', fontweight='bold')
    ax.set_xlim(0, 4)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.1, bar.get_y() + bar.get_height()/2., f'{int(w)} datasets',
                ha='left', va='center', fontsize=12, fontweight='bold')
    ax.text(0.02, 0.98, 'Original: Msoud, NeuroMRI\nExtended: Msoud, NeuroMRI, Epic',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure8_dataset_coverage.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved figure8_dataset_coverage")


def plot_fig9_capabilities():
    fig, ax = plt.subplots(figsize=(12, 7))
    capabilities = ['Tumor\nClassification', 'WHO Tumor\nGrading', 'MC Dropout\nUncertainty', 'Multi-XAI\nExplanations']
    original_scores = [1, 0, 0, 0]
    extended_scores = [1, 1, 1, 1]
    x = np.arange(len(capabilities))
    width = 0.35
    bars1 = ax.bar(x - width/2, original_scores, width, label='Original Paper',
                   color=WORK_COLORS['Original Paper'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, extended_scores, width, label='Extended Work',
                   color=WORK_COLORS['Extended Work'], edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Capability Present (1=Yes, 0=No)', fontweight='bold')
    ax.set_title('Figure 9: Capability Comparison - Original vs Extended Work', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(capabilities)
    ax.set_ylim(0, 1.3)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.03, 'Yes' if h > 0.5 else 'No',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure9_capability_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved figure9_capability_comparison")


def generate_all_plots():
    print(f"\n{'='*70}")
    print("  Generating Comparison Tables and Figures")
    print(f"{'='*70}")
    results = load_all_results()
    if not results['original'] and not results['extended']:
        print("  No results found to plot!")
        return
    df = create_dataframe(results)
    plot_table1(df)
    plot_fig1_accuracy(results)
    plot_fig2_f1(results)
    plot_fig3_precision_recall(results)
    plot_fig4_all_metrics(results)
    plot_fig5_grading(results)
    plot_fig6_radar(results)
    plot_fig7_performance_change(results)
    plot_fig8_dataset_coverage(results)
    plot_fig9_capabilities()
    generate_summary_report(results, df)
    print(f"\n  All comparison outputs saved to: {OUTPUT_DIR}")


def generate_summary_report(results, df):
    report = []
    report.append("=" * 80)
    report.append("COMPARISON REPORT: ORIGINAL PAPER vs EXTENDED WORK")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    report.append("1. SCOPE COMPARISON")
    report.append("-" * 60)
    report.append("Original Paper:")
    report.append("  - Datasets: Msoud, NeuroMRI (2 datasets)")
    report.append("  - Model: Single-task CNN for tumor classification")
    report.append("  - Output: Tumor type (4 classes)")
    report.append("")
    report.append("Extended Work:")
    report.append("  - Datasets: Msoud, NeuroMRI, Epic (3 datasets)")
    report.append("  - Model: Multi-task CNN (classification + WHO tumor grading)")
    report.append("  - Output: Tumor type (4 classes) + WHO Grade (I-IV)")
    report.append("  - Additional: MC Dropout uncertainty, Multi-XAI explanations")
    report.append("")
    report.append("2. CLASSIFICATION PERFORMANCE COMPARISON")
    report.append("-" * 60)
    report.append(f"{'Dataset':<12} {'Work':<18} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    report.append("-" * 80)
    for _, row in df.iterrows():
        report.append(f"{row['Dataset']:<12} {row['Work']:<18} {row['Accuracy']:<12.4f} {row['Precision']:<12.4f} {row['Recall']:<12.4f} {row['F1']:<12.4f}")
    report.append("")
    report.append("3. PERFORMANCE CHANGE (Extended vs Original)")
    report.append("-" * 60)
    for ds in ORIGINAL_DATASETS:
        if ds in results['original'] and ds in results['extended']:
            orig_f1 = results['original'][ds]['f1']
            ext_f1 = results['extended'][ds]['classification']['f1']
            change = ((ext_f1 - orig_f1) / orig_f1) * 100
            report.append(f"  {ds}: F1 change = {change:+.1f}% (Original: {orig_f1:.4f}, Extended: {ext_f1:.4f})")
    report.append("")
    report.append("4. NOVEL CAPABILITIES IN EXTENDED WORK")
    report.append("-" * 60)
    report.append("WHO Tumor Grading (Grade I-IV):")
    for ds in EXTENDED_DATASETS:
        if ds in results['extended']:
            g = results['extended'][ds]['grading']
            report.append(f"  {ds}: Acc={g['accuracy']:.4f}, F1={g['f1']:.4f}")
    report.append("")
    report.append("5. KEY FINDINGS")
    report.append("-" * 60)
    report.append("- Extended work evaluates on 3 datasets vs 2 in original paper")
    report.append("- Extended work adds novel WHO tumor grading capability")
    report.append("- Extended work integrates MC Dropout for uncertainty estimation")
    report.append("- Extended work provides multi-XAI explanation fusion")
    report.append("=" * 80)
    report_text = "\n".join(report)
    with open(OUTPUT_DIR / 'comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("  Saved comparison_report.txt")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("COMPARISON: ORIGINAL PAPER vs EXTENDED WORK")
    print("=" * 80)
    print(f"Configuration: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}")
    print("=" * 80)

    # Run original paper baseline (Msoud + NeuroMRI)
    original_results = []
    for ds in ORIGINAL_DATASETS:
        result = run_original_single_task(ds)
        original_results.append(result)

    # Run extended multi-task (all 3 datasets)
    extended_results = []
    for ds in EXTENDED_DATASETS:
        result = run_extended_multitask(ds)
        extended_results.append(result)

    # Generate comparison tables and figures
    generate_all_plots()

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.rglob('*')):
        if f.is_file():
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
