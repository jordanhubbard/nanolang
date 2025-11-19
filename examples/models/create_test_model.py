#!/usr/bin/env python3
"""
Create simple ONNX models for testing nanolang ONNX module.

Requirements:
    pip install torch onnx

Usage:
    python create_test_model.py
"""

import torch
import torch.nn as nn
import os

class SimpleModel(nn.Module):
    """Simple 2-layer feedforward network"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TinyClassifier(nn.Module):
    """Tiny image classifier (32x32 grayscale)"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def export_simple_model():
    """Export simple feedforward model"""
    print("Creating simple model...")
    model = SimpleModel()
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 10)
    
    # Export to ONNX
    output_path = "simple_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"}
        },
        opset_version=11
    )
    
    print(f"✓ Exported to {output_path}")
    print(f"  Input shape: [batch, 10]")
    print(f"  Output shape: [batch, 5]")
    return output_path

def export_tiny_classifier():
    """Export tiny image classifier"""
    print("\nCreating tiny classifier...")
    model = TinyClassifier(num_classes=10)
    model.eval()
    
    # Create dummy input (batch=1, channels=1, height=32, width=32)
    dummy_input = torch.randn(1, 1, 32, 32)
    
    # Export to ONNX
    output_path = "tiny_classifier.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "logits": {0: "batch"}
        },
        opset_version=11
    )
    
    print(f"✓ Exported to {output_path}")
    print(f"  Input shape: [batch, 1, 32, 32]")
    print(f"  Output shape: [batch, 10]")
    return output_path

def verify_model(model_path):
    """Verify ONNX model"""
    try:
        import onnx
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(f"✓ Model verified: {model_path}")
        return True
    except ImportError:
        print("  (Install 'onnx' package to verify models: pip install onnx)")
        return False
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False

def main():
    print("=" * 60)
    print("ONNX Test Model Generator")
    print("=" * 60)
    print()
    
    # Create models
    models = []
    models.append(export_simple_model())
    models.append(export_tiny_classifier())
    
    print()
    print("=" * 60)
    print("Verifying models...")
    print("=" * 60)
    
    for model_path in models:
        verify_model(model_path)
    
    print()
    print("=" * 60)
    print("Test models created successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run nanolang examples:")
    print("     ./bin/nanoc examples/40_onnx_simple.nano -o onnx_simple")
    print("     ./onnx_simple")
    print()
    print("  2. Try inference:")
    print("     ./bin/nanoc examples/41_onnx_inference.nano -o onnx_inference")
    print("     ./onnx_inference")
    print()

if __name__ == "__main__":
    main()

