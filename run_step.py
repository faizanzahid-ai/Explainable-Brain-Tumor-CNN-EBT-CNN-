"""
Run individual steps or the complete pipeline.
Usage:
  python run_step.py --original Msoud          # Run original baseline on one dataset
  python run_step.py --extended Epic           # Run extended multi-task on one dataset
  python run_step.py --plots                   # Generate comparison plots only
  python run_step.py --model-compare           # Generate model-to-model comparison images
  python run_step.py --pdf                     # Generate PDF report
  python run_step.py --all                     # Run training + plots + comparison + PDF
  python run_step.py --complete                # Run EVERYTHING (full pipeline)
"""

import argparse
import sys
import subprocess

from run_comparison_experiments import (
    run_original_single_task,
    run_extended_multitask,
    generate_all_plots,
    ORIGINAL_DATASETS,
    EXTENDED_DATASETS,
)


def run_script(script_name):
    """Run a Python script in the same environment."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)
    result = subprocess.run([sys.executable, str(script_name)])
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run individual steps or the full pipeline")
    parser.add_argument("--original", metavar="DATASET", choices=["Msoud", "NeuroMRI"],
                        help="Run original single-task baseline on a dataset")
    parser.add_argument("--extended", metavar="DATASET", choices=["Msoud", "Epic", "NeuroMRI"],
                        help="Run extended multi-task on a dataset")
    parser.add_argument("--plots", action="store_true",
                        help="Generate all comparison tables and figures")
    parser.add_argument("--model-compare", action="store_true",
                        help="Generate model-to-model comparison images")
    parser.add_argument("--pdf", action="store_true",
                        help="Generate PDF report")
    parser.add_argument("--all", action="store_true",
                        help="Run training + all plots (same as run_comparison_experiments.py)")
    parser.add_argument("--complete", action="store_true",
                        help="Run FULL pipeline: training + plots + model comparison + PDF report")
    args = parser.parse_args()

    if args.original:
        result = run_original_single_task(args.original)
        sys.exit(0 if result else 1)

    if args.extended:
        result = run_extended_multitask(args.extended)
        sys.exit(0 if result else 1)

    if args.plots:
        generate_all_plots()
        sys.exit(0)

    if args.model_compare:
        success = run_script("model_comparison.py")
        sys.exit(0 if success else 1)

    if args.pdf:
        success = run_script("generate_report_pdf.py")
        sys.exit(0 if success else 1)

    if args.all:
        print("\n[STEP 1/3] Training original baselines...")
        for ds in ORIGINAL_DATASETS:
            run_original_single_task(ds)
        print("\n[STEP 2/3] Training extended multi-task models...")
        for ds in EXTENDED_DATASETS:
            run_extended_multitask(ds)
        print("\n[STEP 3/3] Generating comparison plots...")
        generate_all_plots()
        print("\nAll steps completed!")
        sys.exit(0)

    if args.complete:
        print("\n" + "="*60)
        print("RUNNING COMPLETE PIPELINE")
        print("="*60)

        print("\n[1/6] Training original baselines...")
        for ds in ORIGINAL_DATASETS:
            run_original_single_task(ds)

        print("\n[2/6] Training extended multi-task models...")
        for ds in EXTENDED_DATASETS:
            run_extended_multitask(ds)

        print("\n[3/6] Generating comparison plots...")
        generate_all_plots()

        print("\n[4/6] Generating model-to-model comparison images...")
        run_script("model_comparison.py")

        print("\n[5/6] Generating PDF report...")
        run_script("generate_report_pdf.py")

        print("\n[6/6] Pipeline complete! Check outputs:")
        print("  - comparison_results/  (plots, tables, models)")
        print("  - Extended_Brain_Tumor_Report.pdf")
        print("="*60)
        sys.exit(0)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
