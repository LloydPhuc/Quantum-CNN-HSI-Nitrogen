#!/usr/bin/env python
"""
CLI Entry Point for QCNN Pipeline.

Usage:
    python -m scripts.run_pipeline --data_path ... --label_path ... --wave_path ...
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qcnn import run_qcnn_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run QCNN Hybrid Quantum-Classical Pipeline for HSI Nitrogen Classification"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to reflectance .npy file (260 bands)"
    )
    parser.add_argument(
        "--label_path",
        type=str,
        required=True,
        help="Path to labels .npy file"
    )
    parser.add_argument(
        "--wave_path",
        type=str,
        default=None,
        help="Path to wavelengths .npy file (optional)"
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default="results/features",
        help="Directory to export extracted features"
    )
    parser.add_argument(
        "--vqc_train_size",
        type=int,
        default=None,
        help="Subsample size for VQC training (None = full dataset)"
    )
    args = parser.parse_args()

    print("🚀 Starting QCNN Pipeline...")
    print(f"   Data: {args.data_path}")
    print(f"   Labels: {args.label_path}")
    print(f"   Export: {args.export_dir}")

    results = run_qcnn_pipeline(
        path_X=args.data_path,
        path_y=args.label_path,
        path_wave=args.wave_path,
        export_dir=args.export_dir,
    )

    print("\n✅ Pipeline completed successfully!")
    print(f"   Classical Acc: {results['classical']['results']['accuracy']:.4f}")
    print(f"   Quantum Acc:   {results['vqc']['results']['accuracy']:.4f}")


if __name__ == "__main__":
    main()