#!/usr/bin/env python3
"""
ONNX Model Converter for Browser Compatibility

This script converts ONNX models to be compatible with onnxruntime-web
by downgrading to opset 14 and simplifying the model.

Usage:
    python MODEL_CONVERTER.py input.onnx output.onnx

Requirements:
    pip install onnx onnxruntime onnxsim
"""

import sys
import onnx
from onnx import version_converter
import onnxsim

def convert_model(input_path: str, output_path: str, target_opset: int = 14):
    """
    Convert ONNX model to browser-compatible version

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save converted model
        target_opset: Target opset version (default: 14)
    """
    print(f"Loading model from {input_path}...")
    model = onnx.load(input_path)

    # Check current opset
    current_opset = model.opset_import[0].version
    print(f"Current opset version: {current_opset}")

    if current_opset > target_opset:
        print(f"Converting to opset {target_opset}...")
        model = version_converter.convert_version(model, target_opset)

    # Simplify model
    print("Simplifying model...")
    try:
        model, check = onnxsim.simplify(
            model,
            check_n=3,
            skip_fuse_bn=False,
            skip_optimization=False
        )
        if check:
            print("✅ Model simplified successfully")
        else:
            print("⚠️  Simplification may have issues, but continuing...")
    except Exception as e:
        print(f"⚠️  Could not simplify model: {e}")
        print("Continuing with non-simplified model...")

    # Check the model
    print("Validating model...")
    try:
        onnx.checker.check_model(model)
        print("✅ Model is valid")
    except Exception as e:
        print(f"⚠️  Model validation warning: {e}")

    # Save
    print(f"Saving converted model to {output_path}...")
    onnx.save(model, output_path)

    # Compare sizes
    import os
    input_size = os.path.getsize(input_path) / 1024 / 1024
    output_size = os.path.getsize(output_path) / 1024 / 1024

    print(f"\n{'='*50}")
    print(f"✅ Conversion complete!")
    print(f"Input size:  {input_size:.2f} MB")
    print(f"Output size: {output_size:.2f} MB")
    print(f"Target opset: {target_opset}")
    print(f"{'='*50}\n")

    print("Next steps:")
    print(f"1. Test the converted model: {output_path}")
    print(f"2. Upload to your repository")
    print(f"3. Update the model URL in your app")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python MODEL_CONVERTER.py input.onnx output.onnx [opset_version]")
        print("\nExample:")
        print("  python MODEL_CONVERTER.py detection-v1.0.0.onnx detection-v1.0.0-web.onnx 14")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    target_opset = int(sys.argv[3]) if len(sys.argv) > 3 else 14

    try:
        convert_model(input_path, output_path, target_opset)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
