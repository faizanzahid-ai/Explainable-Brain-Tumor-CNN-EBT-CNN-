"""
Comprehensive experiment runner for Brain Tumor XAI Extension.

Experiments:
  E1: Baseline (single-task) on Msoud
  E2: Baseline (single-task) on Epic & CSCR Hospital
  E3: Multi-task model on Msoud
  E4: Multi-task model on Epic & CSCR Hospital
  E5-E6: Same as E3-E4 + MC Dropout uncertainty
  E7-E8: XAI benchmarking on both datasets

Usage:
  python run_experiments.py --epochs 40 --batch_size 40
  python run_experiments.py --quick   # reduced epochs for fast testing
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from scipy import stats as scipy_stats

from model import create_model, create_multitask_model, MCDropoutModel
from preprocessing import BrainMRIPreprocessor, load_and_preprocess_dataset
from train import BrainTumorTrainer, assign_grade_labels, WHO_GRADES
from explainability import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, IntegratedGradientsExplainer,
    RISEExplainer, XAIFusion, XAIBenchmark,
    visualize_multi_xai, visualize_fusion,
    visualize_uncertainty, visualize_grading_uncertainty
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS = {
    'Msoud': {
        'train_dir': r"c:\Users\zahid\Downloads\ANN Project part2\Msoud\Training",
        'test_dir':  r"c:\Users\zahid\Downloads\ANN Project part2\Msoud\Testing",
        'data_dir':  r"c:\Users\zahid\Downloads\ANN Project part2\Msoud",
    },
    'Epic': {
        'train_dir': r"c:\Users\zahid\Downloads\ANN Project part2\Epic and CSCR hospital Dataset\Epic and CSCR hospital Dataset\Train",
        'test_dir':  r"c:\Users\zahid\Downloads\ANN Project part2\Epic and CSCR hospital Dataset\Epic and CSCR hospital Dataset\Test",
        'data_dir':  r"c:\Users\zahid\Downloads\ANN Project part2\Epic and CSCR hospital Dataset\Epic and CSCR hospital Dataset",
    },
    'NeuroMRI': {
        'train_dir': r"c:\Users\zahid\Downloads\ANN Project part2\NeuroMRI\Training",
        'test_dir':  r"c:\Users\zahid\Downloads\ANN Project part2\NeuroMRI\Testing",
        'data_dir':  r"c:\Users\zahid\Downloads\ANN Project part2\NeuroMRI",
    },
}

CLASSES = ['Meningioma', 'No Tumor', 'Glioma', 'Pituitary']
MC_PASSES = 30
XAI_NUM_SAMPLES = 5
BENCHMARK_NUM_SAMPLES = 3


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

def load_dataset(dataset_name, preprocessor):
    """Load train/val/test splits for a dataset."""
    ds = DATASETS[dataset_name]
    from sklearn.model_selection import train_test_split

    train_images, train_labels, _ = load_and_preprocess_dataset(
        ds['train_dir'], preprocessor, classes=CLASSES
    )
    test_images, test_labels, _ = load_and_preprocess_dataset(
        ds['test_dir'], preprocessor, classes=CLASSES
    )

    # Normalise
    X_train = train_images.astype('float16') / 255.0
    X_test = test_images.astype('float16') / 255.0

    # Validation split (10% of train)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )

    # Grade labels
    y_grade_train = assign_grade_labels(y_train, CLASSES, random_state=42)
    y_grade_val = assign_grade_labels(y_val, CLASSES, random_state=43)
    y_grade_test = assign_grade_labels(test_labels, CLASSES, random_state=44)

    print(f"[{dataset_name}] Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': test_labels,
        'y_grade_train': y_grade_train, 'y_grade_val': y_grade_val, 'y_grade_test': y_grade_test,
    }


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_baseline_experiment(dataset_name, data, output_dir, epochs, batch_size, lr):
    """E1/E2: Single-task baseline."""
    exp_name = f"E_baseline_{dataset_name}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Experiment: {exp_name} (Single-task Baseline)")
    print(f"{'='*70}")

    model = create_model(input_shape=(224, 224, 3), num_classes=4, learning_rate=lr)

    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        exp_dir / 'best_model.keras', monitor='val_accuracy',
        save_best_only=True, mode='max', verbose=0
    )
    cb_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=0
    )
    cb_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0
    )

    t0 = time.time()
    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=epochs, batch_size=batch_size,
        callbacks=[cb_checkpoint, cb_early, cb_lr], verbose=1
    )
    train_time = time.time() - t0

    # Evaluate
    y_pred_probs = model.predict(data['X_test'], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = data['y_test']

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\n  Results: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Baseline Confusion Matrix — {dataset_name}')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(exp_dir / 'confusion_matrix.png', dpi=150)
    plt.close()

    # Training curves
    _plot_training_history(history, exp_dir / 'training_curves.png', dataset_name, 'Baseline')

    # Save results
    results = {
        'experiment': exp_name, 'dataset': dataset_name, 'model': 'Baseline (single-task)',
        'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1),
        'training_time_sec': float(train_time), 'epochs_trained': len(history.history['loss']),
    }
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Save model
    model.save(exp_dir / 'final_model.keras')

    return results, model, y_pred


def run_multitask_experiment(dataset_name, data, output_dir, epochs, batch_size, lr):
    """E3/E4: Multi-task model (classification + grading)."""
    exp_name = f"E_multitask_{dataset_name}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Experiment: {exp_name} (Multi-task)")
    print(f"{'='*70}")

    model = create_multitask_model(input_shape=(224, 224, 3), num_classes=4, num_grades=4, learning_rate=lr)

    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        exp_dir / 'best_model.keras', monitor='val_classification_accuracy',
        save_best_only=True, mode='max', verbose=0
    )
    cb_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_classification_accuracy', patience=10, restore_best_weights=True, mode='max', verbose=0
    )
    cb_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0
    )

    train_data = {'classification': data['y_train'], 'grading': data['y_grade_train']}
    val_data = (data['X_val'], {'classification': data['y_val'], 'grading': data['y_grade_val']})

    t0 = time.time()
    history = model.fit(
        data['X_train'], train_data,
        validation_data=val_data,
        epochs=epochs, batch_size=batch_size,
        callbacks=[cb_checkpoint, cb_early, cb_lr], verbose=1
    )
    train_time = time.time() - t0

    # Evaluate
    y_pred_cls, y_pred_grade = model.predict(data['X_test'], verbose=0)
    y_pred_cls_idx = np.argmax(y_pred_cls, axis=1)
    y_pred_grade_idx = np.argmax(y_pred_grade, axis=1)
    y_true = data['y_test']
    y_grade_true = data['y_grade_test']

    cls_acc = accuracy_score(y_true, y_pred_cls_idx)
    cls_prec = precision_score(y_true, y_pred_cls_idx, average='weighted')
    cls_rec = recall_score(y_true, y_pred_cls_idx, average='weighted')
    cls_f1 = f1_score(y_true, y_pred_cls_idx, average='weighted')

    grade_acc = accuracy_score(y_grade_true, y_pred_grade_idx)
    grade_prec = precision_score(y_grade_true, y_pred_grade_idx, average='weighted')
    grade_rec = recall_score(y_grade_true, y_pred_grade_idx, average='weighted')
    grade_f1 = f1_score(y_grade_true, y_pred_grade_idx, average='weighted')

    print(f"\n  Classification: Acc={cls_acc:.4f}, Prec={cls_prec:.4f}, Rec={cls_rec:.4f}, F1={cls_f1:.4f}")
    print(f"  Grading:        Acc={grade_acc:.4f}, Prec={grade_prec:.4f}, Rec={grade_rec:.4f}, F1={grade_f1:.4f}")

    # Confusion matrices
    cm_cls = confusion_matrix(y_true, y_pred_cls_idx)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_cls, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Multi-task Classification CM — {dataset_name}')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(exp_dir / 'classification_cm.png', dpi=150)
    plt.close()

    cm_grade = confusion_matrix(y_grade_true, y_pred_grade_idx)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_grade, annot=True, fmt='d', cmap='Blues', xticklabels=WHO_GRADES, yticklabels=WHO_GRADES)
    plt.title(f'WHO Grading CM — {dataset_name}')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(exp_dir / 'grading_cm.png', dpi=150)
    plt.close()

    # Training curves
    _plot_training_history(history, exp_dir / 'training_curves.png', dataset_name, 'Multi-task')

    results = {
        'experiment': exp_name, 'dataset': dataset_name, 'model': 'Multi-task (cls+grading)',
        'classification': {
            'accuracy': float(cls_acc), 'precision': float(cls_prec),
            'recall': float(cls_rec), 'f1': float(cls_f1),
        },
        'grading': {
            'accuracy': float(grade_acc), 'precision': float(grade_prec),
            'recall': float(grade_rec), 'f1': float(grade_f1),
        },
        'training_time_sec': float(train_time), 'epochs_trained': len(history.history['loss']),
    }
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)

    model.save(exp_dir / 'final_model.keras')
    return results, model, y_pred_cls_idx


def run_uncertainty_experiment(dataset_name, data, model, output_dir):
    """E5/E6: MC Dropout uncertainty estimation."""
    exp_name = f"E_uncertainty_{dataset_name}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Experiment: {exp_name} (MC Dropout Uncertainty)")
    print(f"{'='*70}")

    mc_model = MCDropoutModel(model, num_passes=MC_PASSES)
    is_multitask = isinstance(model.output, list)

    if is_multitask:
        uncertainty = mc_model.predict_with_uncertainty_multitask(data['X_test'])
        cls_unc = uncertainty['classification']
        grade_unc = uncertainty['grading']

        # Correlation: confidence vs correctness
        y_pred_cls = cls_unc['pred_class']
        y_true = data['y_test']
        correct = (y_pred_cls == y_true).astype(float)
        corr, p_val = scipy_stats.pointbiserialr(correct, cls_unc['confidence'])

        # ECE (Expected Calibration Error)
        ece = _compute_ece(cls_unc['confidence'], correct, n_bins=10)

        print(f"\n  Classification Uncertainty:")
        print(f"    Mean confidence: {np.mean(cls_unc['confidence']):.4f}")
        print(f"    Mean entropy: {np.mean(cls_unc['entropy']):.4f}")
        print(f"    Mean epistemic (MI): {np.mean(cls_unc['mutual_info']):.4f}")
        print(f"    Confidence-correctness correlation: r={corr:.4f}, p={p_val:.6f}")
        print(f"    ECE: {ece:.4f}")

        print(f"\n  Grading Uncertainty:")
        print(f"    Mean confidence: {np.mean(grade_unc['confidence']):.4f}")
        print(f"    Mean entropy: {np.mean(grade_unc['entropy']):.4f}")
        print(f"    Mean epistemic (MI): {np.mean(grade_unc['mutual_info']):.4f}")

        # Visualise
        _plot_uncertainty_distribution(cls_unc, correct, exp_dir / 'cls_uncertainty_dist.png', 'Classification')
        _plot_uncertainty_distribution(grade_unc, None, exp_dir / 'grade_uncertainty_dist.png', 'Grading')

        # Sample visualisations
        unc_vis_dir = exp_dir / 'uncertainty_samples'
        unc_vis_dir.mkdir(exist_ok=True)
        # Use indices within the uncertainty results size
        n_samples = min(XAI_NUM_SAMPLES, len(cls_unc['pred_class']))
        indices = np.random.choice(len(cls_unc['pred_class']), n_samples, replace=False)
        
        # Extract only scalar metrics for visualization (skip multi-dimensional arrays)
        cls_unc_scalar = {k: v for k, v in cls_unc.items() if not isinstance(v, np.ndarray) or v.ndim == 1}
        grade_unc_scalar = {k: v for k, v in grade_unc.items() if not isinstance(v, np.ndarray) or v.ndim == 1}
        
        for i, idx in enumerate(indices):
            img255 = (data['X_test'][idx] * 255).astype(np.uint8)
            cls_name = CLASSES[y_pred_cls[idx]]
            grade_idx = grade_unc['pred_class'][idx]
            grade_name = WHO_GRADES[grade_idx]
            
            # Get sample-specific values
            cls_sample = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in cls_unc_scalar.items()}
            grade_sample = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in grade_unc_scalar.items()}
            
            visualize_grading_uncertainty(
                img255, cls_sample, grade_sample,
                cls_name, grade_name, save_path=unc_vis_dir / f'uncertainty_{i}.png'
            )

        results = {
            'experiment': exp_name, 'dataset': dataset_name,
            'classification': {
                'mean_confidence': float(np.mean(cls_unc['confidence'])),
                'mean_entropy': float(np.mean(cls_unc['entropy'])),
                'mean_epistemic_mi': float(np.mean(cls_unc['mutual_info'])),
                'mean_variation_ratio': float(np.mean(cls_unc['variation_ratio'])),
                'confidence_correctness_r': float(corr),
                'confidence_correctness_p': float(p_val),
                'ece': float(ece),
            },
            'grading': {
                'mean_confidence': float(np.mean(grade_unc['confidence'])),
                'mean_entropy': float(np.mean(grade_unc['entropy'])),
                'mean_epistemic_mi': float(np.mean(grade_unc['mutual_info'])),
                'mean_variation_ratio': float(np.mean(grade_unc['variation_ratio'])),
            }
        }
    else:
        uncertainty = mc_model.predict_with_uncertainty(data['X_test'])
        y_pred = uncertainty['pred_class']
        y_true = data['y_test']
        correct = (y_pred == y_true).astype(float)
        corr, p_val = scipy_stats.pointbiserialr(correct, uncertainty['confidence'])
        ece = _compute_ece(uncertainty['confidence'], correct, n_bins=10)

        print(f"\n  Uncertainty:")
        print(f"    Mean confidence: {np.mean(uncertainty['confidence']):.4f}")
        print(f"    Mean entropy: {np.mean(uncertainty['entropy']):.4f}")
        print(f"    Mean epistemic (MI): {np.mean(uncertainty['mutual_info']):.4f}")
        print(f"    Confidence-correctness r={corr:.4f}, p={p_val:.6f}")
        print(f"    ECE: {ece:.4f}")

        _plot_uncertainty_distribution(uncertainty, correct, exp_dir / 'uncertainty_dist.png', 'Classification')

        results = {
            'experiment': exp_name, 'dataset': dataset_name,
            'mean_confidence': float(np.mean(uncertainty['confidence'])),
            'mean_entropy': float(np.mean(uncertainty['entropy'])),
            'mean_epistemic_mi': float(np.mean(uncertainty['mutual_info'])),
            'mean_variation_ratio': float(np.mean(uncertainty['variation_ratio'])),
            'confidence_correctness_r': float(corr),
            'confidence_correctness_p': float(p_val),
            'ece': float(ece),
        }

    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results


def run_xai_benchmark(dataset_name, data, model, y_pred, output_dir):
    """E7/E8: XAI benchmarking (deletion/insertion/consistency)."""
    exp_name = f"E_xai_benchmark_{dataset_name}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Experiment: {exp_name} (XAI Benchmark)")
    print(f"{'='*70}")

    explainers = {
        'Grad-CAM': GradCAM(model),
        'Grad-CAM++': GradCAMPlusPlus(model),
        'Score-CAM': ScoreCAM(model, num_masks=64),
        'Integrated-Grad': IntegratedGradientsExplainer(model, num_steps=30),
        'RISE': RISEExplainer(model, num_masks=200, mask_resolution=8),
    }

    benchmark = XAIBenchmark(model, explainers, num_steps=15)

    indices = np.random.choice(len(data['X_test']), min(BENCHMARK_NUM_SAMPLES, len(data['X_test'])), replace=False)

    all_results = {}
    for idx in indices:
        image = data['X_test'][idx]
        pred_label = y_pred[idx]
        print(f"  Benchmarking sample {idx} (class: {CLASSES[pred_label]})...")
        results = benchmark.benchmark(image, pred_label)
        all_results[idx] = results

    # Aggregate
    print(f"\n  Aggregated XAI Benchmark — {dataset_name}:")
    print(f"  {'Method':<20} {'Del-AUC':<12} {'Ins-AUC':<12} {'Avg-IoU':<12}")
    print(f"  {'-'*56}")

    agg = {}
    for method in explainers.keys():
        del_aucs = [all_results[idx][method]['deletion_auc'] for idx in all_results]
        ins_aucs = [all_results[idx][method]['insertion_auc'] for idx in all_results]
        consistencies = []
        for idx in all_results:
            consistencies.extend(list(all_results[idx][method]['consistency'].values()))

        mean_del = np.mean(del_aucs)
        mean_ins = np.mean(ins_aucs)
        mean_iou = np.mean(consistencies) if consistencies else 0.0

        agg[method] = {
            'deletion_auc_mean': float(mean_del),
            'insertion_auc_mean': float(mean_ins),
            'consistency_iou_mean': float(mean_iou),
            'deletion_auc_std': float(np.std(del_aucs)),
            'insertion_auc_std': float(np.std(ins_aucs)),
        }
        print(f"  {method:<20} {mean_del:<12.4f} {mean_ins:<12.4f} {mean_iou:<12.4f}")

    # Fusion benchmark
    print(f"\n  Fusion benchmark...")
    fusion_mean = XAIFusion(fusion_method='mean')
    fusion_cons = XAIFusion(fusion_method='consensus')

    for fusion_name, fusion_engine in [('Fusion-Mean', fusion_mean), ('Fusion-Consensus', fusion_cons)]:
        del_aucs, ins_aucs = [], []
        for idx in indices:
            image = data['X_test'][idx]
            pred_label = y_pred[idx]
            heatmaps = [all_results[idx][m]['heatmap'] for m in explainers.keys()]
            fused = fusion_engine.fuse(heatmaps)
            del_aucs.append(benchmark.deletion_score(image, fused, pred_label))
            ins_aucs.append(benchmark.insertion_score(image, fused, pred_label))

        agg[fusion_name] = {
            'deletion_auc_mean': float(np.mean(del_aucs)),
            'insertion_auc_mean': float(np.mean(ins_aucs)),
            'deletion_auc_std': float(np.std(del_aucs)),
            'insertion_auc_std': float(np.std(ins_aucs)),
            'consistency_iou_mean': 0.0,
        }
        print(f"  {fusion_name:<20} {np.mean(del_aucs):<12.4f} {np.mean(ins_aucs):<12.4f}")

    # Plot benchmark bar chart
    _plot_xai_benchmark(agg, exp_dir / 'xai_benchmark.png', dataset_name)

    # Multi-XAI sample visualisations
    xai_vis_dir = exp_dir / 'xai_samples'
    xai_vis_dir.mkdir(exist_ok=True)
    for idx in indices:
        image = data['X_test'][idx]
        pred_label = y_pred[idx]
        image_255 = (image * 255).astype(np.uint8)

        heatmaps_dict = {}
        for name, explainer in explainers.items():
            hm, ov = explainer.explain(image, pred_label, alpha=0.4)
            heatmaps_dict[name] = (hm, ov)

        cls_name = CLASSES[pred_label]
        visualize_multi_xai(image_255, heatmaps_dict, cls_name,
                            save_path=xai_vis_dir / f'multi_xai_{idx}.png')

        # Fusion visualisation
        hm_list = [h for h, _ in heatmaps_dict.values()]
        fused = fusion_mean.fuse(hm_list)
        import cv2
        fused_resized = cv2.resize(fused, (image_255.shape[1], image_255.shape[0]))
        fused_uint8 = np.uint8(255 * fused_resized)
        fused_colored = cv2.applyColorMap(fused_uint8, cv2.COLORMAP_JET)
        fused_colored = cv2.cvtColor(fused_colored, cv2.COLOR_BGR2RGB)
        fused_overlay = cv2.addWeighted(image_255, 0.6, fused_colored, 0.4, 0)

        visualize_fusion(image_255, heatmaps_dict, fused, fused_overlay, cls_name,
                         save_path=xai_vis_dir / f'fusion_{idx}.png')

    # Save
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(agg, f, indent=4)

    return agg


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(all_results, output_dir):
    """Run statistical significance tests across experiments."""
    stat_dir = output_dir / 'statistical_tests'
    stat_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Statistical Significance Tests")
    print(f"{'='*70}")

    tests = {}

    # Test 1: Classification accuracy comparison (McNemar's test proxy)
    # We compare per-sample correctness of baseline vs multitask on same dataset
    for ds_name in DATASETS.keys():
        baseline_key = f"E_baseline_{ds_name}"
        mt_key = f"E_multitask_{ds_name}"

        if baseline_key in all_results and mt_key in all_results:
            b_res = all_results[baseline_key]
            m_res = all_results[mt_key]

            # We need per-sample predictions — check if stored
            b_pred = b_res.get('y_pred')
            m_pred = m_res.get('y_pred')
            y_true = b_res.get('y_true')

            if b_pred is not None and m_pred is not None and y_true is not None:
                b_correct = (b_pred == y_true).astype(int)
                m_correct = (m_pred == y_true).astype(int)

                # McNemar's test
                n01 = np.sum((b_correct == 0) & (m_correct == 1))  # baseline wrong, multitask right
                n10 = np.sum((b_correct == 1) & (m_correct == 0))  # baseline right, multitask wrong

                if n01 + n10 > 0:
                    # With continuity correction
                    chi2 = (abs(n01 - n10) - 0.5) ** 2 / (n01 + n10 + 1e-10)
                    p_value = 1 - scipy_stats.chi2.cdf(chi2, df=1)
                else:
                    chi2 = 0.0
                    p_value = 1.0

                # Paired t-test on per-sample correctness
                t_stat, t_pval = scipy_stats.ttest_rel(m_correct, b_correct)

                # Wilcoxon signed-rank (non-parametric)
                try:
                    w_stat, w_pval = scipy_stats.wilcoxon(m_correct - b_correct + 1e-10)
                except:
                    w_stat, w_pval = float('nan'), float('nan')

                test_name = f"Baseline_vs_Multitask_{ds_name}"
                tests[test_name] = {
                    'dataset': ds_name,
                    'baseline_acc': float(accuracy_score(y_true, b_pred)),
                    'multitask_acc': float(accuracy_score(y_true, m_pred)),
                    'delta_acc': float(accuracy_score(y_true, m_pred) - accuracy_score(y_true, b_pred)),
                    'mcnemar_n01': int(n01), 'mcnemar_n10': int(n10),
                    'mcnemar_chi2': float(chi2), 'mcnemar_p': float(p_value),
                    'paired_t_stat': float(t_stat), 'paired_t_p': float(t_pval),
                    'wilcoxon_stat': float(w_stat), 'wilcoxon_p': float(w_pval),
                    'significant_005': bool(p_value < 0.05),
                }

                print(f"\n  {test_name}:")
                print(f"    Baseline Acc: {tests[test_name]['baseline_acc']:.4f}")
                print(f"    Multitask Acc: {tests[test_name]['multitask_acc']:.4f}")
                print(f"    ΔAcc: {tests[test_name]['delta_acc']:+.4f}")
                print(f"    McNemar χ²={chi2:.4f}, p={p_value:.6f}")
                print(f"    Paired t={t_stat:.4f}, p={t_pval:.6f}")
                sig = "YES" if p_value < 0.05 else "NO"
                print(f"    Significant (α=0.05): {sig}")

    # Test 2: XAI method ranking (Friedman test proxy)
    for ds_name in DATASETS.keys():
        xai_key = f"E_xai_benchmark_{ds_name}"
        if xai_key in all_results:
            xai_res = all_results[xai_key]
            methods = [k for k in xai_res.keys() if 'deletion_auc_mean' in xai_res[k]]

            if len(methods) >= 3:
                del_aucs = [xai_res[m]['deletion_auc_mean'] for m in methods]
                ins_aucs = [xai_res[m]['insertion_auc_mean'] for m in methods]

                # Kendall's W (agreement on ranking) — simplified
                del_ranking = np.argsort(np.argsort(del_aucs))  # rank (lower deletion = better)
                ins_ranking = np.argsort(np.argsort(ins_aucs)[::-1])  # rank (higher insertion = better)

                tau, tau_p = scipy_stats.kendalltau(del_ranking, ins_ranking)

                test_name = f"XAI_Ranking_Agreement_{ds_name}"
                tests[test_name] = {
                    'dataset': ds_name,
                    'methods': methods,
                    'deletion_aucs': {m: float(v) for m, v in zip(methods, del_aucs)},
                    'insertion_aucs': {m: float(v) for m, v in zip(methods, ins_aucs)},
                    'kendall_tau': float(tau),
                    'kendall_p': float(tau_p),
                    'ranking_agreement': bool(tau_p < 0.05),
                }

                print(f"\n  {test_name}:")
                print(f"    Kendall τ={tau:.4f}, p={tau_p:.6f}")
                print(f"    Deletion ranking (lower=better): {list(zip(methods, del_aucs))}")
                print(f"    Insertion ranking (higher=better): {list(zip(methods, ins_aucs))}")

    with open(stat_dir / 'statistical_tests.json', 'w') as f:
        json.dump(tests, f, indent=4)

    return tests


# ---------------------------------------------------------------------------
# Final report generation
# ---------------------------------------------------------------------------

def generate_final_report(all_results, stat_tests, output_dir):
    """Generate comprehensive comparison tables and visualisations."""
    report_dir = output_dir / 'final_report'
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Generating Final Report")
    print(f"{'='*70}")

    # --- Table 1: Classification Performance Comparison ---
    table1_data = []
    for ds_name in DATASETS.keys():
        for model_type, prefix in [('Baseline', 'E_baseline'), ('Multi-task', 'E_multitask')]:
            key = f"{prefix}_{ds_name}"
            if key in all_results:
                r = all_results[key]
                if 'accuracy' in r:
                    table1_data.append({
                        'Dataset': ds_name, 'Model': model_type,
                        'Accuracy': r['accuracy'], 'Precision': r['precision'],
                        'Recall': r['recall'], 'F1': r['f1'],
                    })
                elif 'classification' in r:
                    c = r['classification']
                    table1_data.append({
                        'Dataset': ds_name, 'Model': model_type,
                        'Accuracy': c['accuracy'], 'Precision': c['precision'],
                        'Recall': c['recall'], 'F1': c['f1'],
                    })

    if table1_data:
        _plot_performance_table(table1_data, report_dir / 'table1_classification.png')

    # --- Table 2: Grading Performance ---
    table2_data = []
    for ds_name in DATASETS.keys():
        key = f"E_multitask_{ds_name}"
        if key in all_results and 'grading' in all_results[key]:
            g = all_results[key]['grading']
            table2_data.append({
                'Dataset': ds_name, 'Accuracy': g['accuracy'],
                'Precision': g['precision'], 'Recall': g['recall'], 'F1': g['f1'],
            })

    if table2_data:
        _plot_grading_table(table2_data, report_dir / 'table2_grading.png')

    # --- Table 3: XAI Benchmark ---
    for ds_name in DATASETS.keys():
        key = f"E_xai_benchmark_{ds_name}"
        if key in all_results:
            _plot_xai_benchmark(all_results[key], report_dir / f'table3_xai_benchmark_{ds_name}.png', ds_name)

    # --- Table 4: Uncertainty ---
    for ds_name in DATASETS.keys():
        key = f"E_uncertainty_{ds_name}"
        if key in all_results:
            r = all_results[key]
            _plot_uncertainty_table(r, report_dir / f'table4_uncertainty_{ds_name}.png', ds_name)

    # --- Summary text report ---
    with open(report_dir / 'experiment_report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BRAIN TUMOR XAI EXTENSION — EXPERIMENT REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. CLASSIFICATION PERFORMANCE\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Dataset':<12} {'Model':<15} {'Acc':<10} {'Prec':<10} {'Rec':<10} {'F1':<10}\n")
        for row in table1_data:
            f.write(f"{row['Dataset']:<12} {row['Model']:<15} {row['Accuracy']:<10.4f} {row['Precision']:<10.4f} {row['Recall']:<10.4f} {row['F1']:<10.4f}\n")

        if table2_data:
            f.write("\n2. WHO GRADING PERFORMANCE\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Dataset':<12} {'Acc':<10} {'Prec':<10} {'Rec':<10} {'F1':<10}\n")
            for row in table2_data:
                f.write(f"{row['Dataset']:<12} {row['Accuracy']:<10.4f} {row['Precision']:<10.4f} {row['Recall']:<10.4f} {row['F1']:<10.4f}\n")

        f.write("\n3. STATISTICAL TESTS\n")
        f.write("-" * 60 + "\n")
        for name, test in stat_tests.items():
            f.write(f"\n{name}:\n")
            for k, v in test.items():
                if isinstance(v, (int, float, bool)):
                    f.write(f"  {k}: {v}\n")

        f.write("\n4. XAI BENCHMARK\n")
        f.write("-" * 60 + "\n")
        for ds_name in DATASETS.keys():
            key = f"E_xai_benchmark_{ds_name}"
            if key in all_results:
                f.write(f"\n  {ds_name}:\n")
                f.write(f"  {'Method':<20} {'Del-AUC':<12} {'Ins-AUC':<12} {'IoU':<12}\n")
                for method, data in all_results[key].items():
                    f.write(f"  {method:<20} {data.get('deletion_auc_mean',0):<12.4f} {data.get('insertion_auc_mean',0):<12.4f} {data.get('consistency_iou_mean',0):<12.4f}\n")

    print(f"\n  Report saved to {report_dir}")
    return report_dir


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_training_history(history, save_path, dataset_name, model_name):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title(f'{model_name} Loss — {dataset_name}')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    if 'val_classification_accuracy' in history.history:
        axes[1].plot(history.history['classification_accuracy'], label='Train Cls Acc')
        axes[1].plot(history.history['val_classification_accuracy'], label='Val Cls Acc')
    else:
        axes[1].plot(history.history['accuracy'], label='Train Acc')
        axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_title(f'{model_name} Accuracy — {dataset_name}')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_performance_table(data, save_path):
    """Plot classification comparison as table + bar chart."""
    fig, (ax_table, ax_bar) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [1.2, 1]})

    # Table
    cell_text = [[f"{r['Accuracy']:.4f}", f"{r['Precision']:.4f}", f"{r['Recall']:.4f}", f"{r['F1']:.4f}"] for r in data]
    row_labels = [f"{r['Dataset']} / {r['Model']}" for r in data]
    col_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    ax_table.axis('off')
    table = ax_table.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center')
    table.auto_set_font_size(False); table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax_table.set_title('Classification Performance Comparison', fontsize=14, pad=20)

    # Bar chart
    datasets = list(set(r['Dataset'] for r in data))
    x = np.arange(len(datasets))
    width = 0.35
    baseline_f1 = [r['F1'] for r in data if r['Model'] == 'Baseline' and r['Dataset'] in datasets]
    mt_f1 = [r['F1'] for r in data if r['Model'] == 'Multi-task' and r['Dataset'] in datasets]

    # Align by dataset order
    b_f1 = []; m_f1 = []
    for ds in datasets:
        b = [r['F1'] for r in data if r['Model'] == 'Baseline' and r['Dataset'] == ds]
        m = [r['F1'] for r in data if r['Model'] == 'Multi-task' and r['Dataset'] == ds]
        b_f1.append(b[0] if b else 0)
        m_f1.append(m[0] if m else 0)

    ax_bar.bar(x - width/2, b_f1, width, label='Baseline', color='steelblue')
    ax_bar.bar(x + width/2, m_f1, width, label='Multi-task', color='coral')
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(datasets)
    ax_bar.set_ylabel('F1 Score'); ax_bar.set_title('F1 Score Comparison')
    ax_bar.legend(); ax_bar.set_ylim(0, 1.05); ax_bar.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_grading_table(data, save_path):
    """Plot grading performance table."""
    fig, ax = plt.subplots(figsize=(10, 3))
    cell_text = [[f"{r['Accuracy']:.4f}", f"{r['Precision']:.4f}", f"{r['Recall']:.4f}", f"{r['F1']:.4f}"] for r in data]
    row_labels = [r['Dataset'] for r in data]
    col_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center')
    table.auto_set_font_size(False); table.set_fontsize(12)
    table.scale(1.2, 2.0)
    ax.set_title('WHO Tumor Grading Performance (Grade I–IV)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_xai_benchmark(agg, save_path, dataset_name):
    """Plot XAI benchmark as grouped bar chart."""
    methods = list(agg.keys())
    del_aucs = [agg[m]['deletion_auc_mean'] for m in methods]
    ins_aucs = [agg[m]['insertion_auc_mean'] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, del_aucs, width, label='Deletion AUC (↓ better)', color='salmon')
    bars2 = ax.bar(x + width/2, ins_aucs, width, label='Insertion AUC (↑ better)', color='seagreen')

    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_ylabel('AUC'); ax.set_title(f'XAI Benchmark — {dataset_name}')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(del_aucs), max(ins_aucs)) * 1.15)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_uncertainty_distribution(unc, correct, save_path, task_name):
    """Plot uncertainty distributions for correct vs incorrect predictions."""
    if correct is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    confidence = unc['confidence']
    entropy = unc['entropy']
    mutual_info = unc['mutual_info']

    # Confidence distribution
    axes[0].hist(confidence[correct == 1], bins=30, alpha=0.6, label='Correct', color='green')
    axes[0].hist(confidence[correct == 0], bins=30, alpha=0.6, label='Incorrect', color='red')
    axes[0].set_title(f'{task_name} Confidence Distribution')
    axes[0].set_xlabel('Confidence'); axes[0].legend()

    # Entropy distribution
    axes[1].hist(entropy[correct == 1], bins=30, alpha=0.6, label='Correct', color='green')
    axes[1].hist(entropy[correct == 0], bins=30, alpha=0.6, label='Incorrect', color='red')
    axes[1].set_title(f'{task_name} Entropy Distribution')
    axes[1].set_xlabel('Entropy'); axes[1].legend()

    # Epistemic (MI) distribution
    axes[2].hist(mutual_info[correct == 1], bins=30, alpha=0.6, label='Correct', color='green')
    axes[2].hist(mutual_info[correct == 0], bins=30, alpha=0.6, label='Incorrect', color='red')
    axes[2].set_title(f'{task_name} Epistemic Uncertainty (MI)')
    axes[2].set_xlabel('Mutual Information'); axes[2].legend()

    plt.suptitle('Uncertainty: Correct vs Incorrect Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_uncertainty_table(results, save_path, dataset_name):
    """Plot uncertainty metrics as table."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')

    if 'classification' in results:
        cls = results['classification']
        grade = results['grading']
        cell_text = [
            ['Classification', f"{cls['mean_confidence']:.4f}", f"{cls['mean_entropy']:.4f}",
             f"{cls['mean_epistemic_mi']:.4f}", f"{cls.get('ece','N/A')}"],
            ['Grading', f"{grade['mean_confidence']:.4f}", f"{grade['mean_entropy']:.4f}",
             f"{grade['mean_epistemic_mi']:.4f}", 'N/A'],
        ]
    else:
        cell_text = [
            ['Classification', f"{results['mean_confidence']:.4f}", f"{results['mean_entropy']:.4f}",
             f"{results['mean_epistemic_mi']:.4f}", f"{results.get('ece','N/A')}"],
        ]

    col_labels = ['Task', 'Mean Conf', 'Mean Entropy', 'Mean MI (Epistemic)', 'ECE']
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center')
    table.auto_set_font_size(False); table.set_fontsize(12)
    table.scale(1.2, 2.0)
    ax.set_title(f'MC Dropout Uncertainty — {dataset_name}', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _compute_ece(confidences, correct, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correct[mask].mean()
            ece += mask.sum() / len(confidences) * abs(avg_acc - avg_conf)
    return ece


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run all brain tumor XAI experiments')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--quick', action='store_true', help='Quick mode: fewer epochs, smaller XAI samples')
    parser.add_argument('--output_dir', type=str, default='experiment_results', help='Output directory')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, use existing models')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 5
        global XAI_NUM_SAMPLES, BENCHMARK_NUM_SAMPLES, MC_PASSES
        XAI_NUM_SAMPLES = 2
        BENCHMARK_NUM_SAMPLES = 2
        MC_PASSES = 5

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  Brain Tumor XAI Extension — Full Experiment Suite")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*70}")

    preprocessor = BrainMRIPreprocessor(target_size=(224, 224))
    all_results = {}
    trained_models = {}

    # Load datasets
    datasets_data = {}
    for ds_name in DATASETS.keys():
        print(f"\nLoading {ds_name} dataset...")
        datasets_data[ds_name] = load_dataset(ds_name, preprocessor)

    # ---- E1/E2: Baseline experiments ----
    for ds_name in DATASETS.keys():
        if not args.skip_training:
            results, model, y_pred = run_baseline_experiment(
                ds_name, datasets_data[ds_name], output_dir,
                args.epochs, args.batch_size, args.lr
            )
            # Store per-sample predictions for statistical tests
            results['y_pred'] = y_pred
            results['y_true'] = datasets_data[ds_name]['y_test']
            all_results[f"E_baseline_{ds_name}"] = results
            trained_models[f"baseline_{ds_name}"] = model
        else:
            # Load existing
            model_path = output_dir / f"E_baseline_{ds_name}" / "final_model.keras"
            if model_path.exists():
                model = tf.keras.models.load_model(model_path)
                y_pred_probs = model.predict(datasets_data[ds_name]['X_test'], verbose=0)
                y_pred = np.argmax(y_pred_probs, axis=1)
                y_true = datasets_data[ds_name]['y_test']
                results = {
                    'y_pred': y_pred, 'y_true': y_true,
                    'accuracy': float(accuracy_score(y_true, y_pred)),
                    'precision': float(precision_score(y_true, y_pred, average='weighted')),
                    'recall': float(recall_score(y_true, y_pred, average='weighted')),
                    'f1': float(f1_score(y_true, y_pred, average='weighted')),
                }
                all_results[f"E_baseline_{ds_name}"] = results
                trained_models[f"baseline_{ds_name}"] = model

    # ---- E3/E4: Multi-task experiments ----
    for ds_name in DATASETS.keys():
        if not args.skip_training:
            results, model, y_pred = run_multitask_experiment(
                ds_name, datasets_data[ds_name], output_dir,
                args.epochs, args.batch_size, args.lr
            )
            results['y_pred'] = y_pred
            results['y_true'] = datasets_data[ds_name]['y_test']
            all_results[f"E_multitask_{ds_name}"] = results
            trained_models[f"multitask_{ds_name}"] = model
        else:
            model_path = output_dir / f"E_multitask_{ds_name}" / "final_model.keras"
            if model_path.exists():
                model = tf.keras.models.load_model(model_path)
                y_pred_cls, _ = model.predict(datasets_data[ds_name]['X_test'], verbose=0)
                y_pred = np.argmax(y_pred_cls, axis=1)
                y_true = datasets_data[ds_name]['y_test']
                results = {
                    'y_pred': y_pred, 'y_true': y_true,
                    'classification': {
                        'accuracy': float(accuracy_score(y_true, y_pred)),
                        'precision': float(precision_score(y_true, y_pred, average='weighted')),
                        'recall': float(recall_score(y_true, y_pred, average='weighted')),
                        'f1': float(f1_score(y_true, y_pred, average='weighted')),
                    },
                    'grading': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
                }
                all_results[f"E_multitask_{ds_name}"] = results
                trained_models[f"multitask_{ds_name}"] = model

    # ---- E5/E6: Uncertainty experiments ----
    for ds_name in DATASETS.keys():
        model_key = f"multitask_{ds_name}"
        if model_key in trained_models:
            results = run_uncertainty_experiment(
                ds_name, datasets_data[ds_name], trained_models[model_key], output_dir
            )
            all_results[f"E_uncertainty_{ds_name}"] = results

    # ---- E7/E8: XAI Benchmark ----
    for ds_name in DATASETS.keys():
        model_key = f"multitask_{ds_name}"
        if model_key in trained_models:
            mt_key = f"E_multitask_{ds_name}"
            y_pred = all_results[mt_key].get('y_pred', np.zeros(len(datasets_data[ds_name]['y_test']), dtype=int))
            results = run_xai_benchmark(
                ds_name, datasets_data[ds_name], trained_models[model_key], y_pred, output_dir
            )
            all_results[f"E_xai_benchmark_{ds_name}"] = results

    # ---- Statistical tests ----
    stat_tests = run_statistical_tests(all_results, output_dir)

    # ---- Final report ----
    report_dir = generate_final_report(all_results, stat_tests, output_dir)

    # Save all results
    # Remove non-serializable keys (numpy arrays)
    clean_results = {}
    for k, v in all_results.items():
        clean = {}
        for kk, vv in v.items():
            if isinstance(vv, np.ndarray):
                continue  # skip arrays
            clean[kk] = vv
        clean_results[k] = clean

    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(clean_results, f, indent=4)

    print(f"\n{'#'*70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Results: {output_dir}")
    print(f"  Report:  {report_dir}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
