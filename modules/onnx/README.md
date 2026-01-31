# ONNX Runtime Module for nanolang

CPU-based neural network inference using [ONNX Runtime](https://onnxruntime.ai/).

## Features

- ✅ Load pre-trained ONNX models
- ✅ Run CPU-based inference (no GPU required)
- ✅ Support for models from PyTorch, TensorFlow, scikit-learn
- ✅ Clean C API wrapped for nanolang
- ✅ Automatic memory management

## Installation

### macOS
```bash
brew install onnxruntime
```

### Linux
```bash
sudo apt-get install libonnxruntime-dev
```

### Verify Installation
```bash
pkg-config --exists onnxruntime && echo "ONNX Runtime installed" || echo "Not found"
```

## Quick Start

### 1. Create a Simple ONNX Model (Python)

```python
import torch
import torch.nn as nn

# Simple 2-layer neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create and export model
model = SimpleModel()
model.eval()

dummy_input = torch.randn(1, 10)
torch.onnx.export(
    model,
    dummy_input,
    "simple_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

print("Model exported to simple_model.onnx")
```

### 2. Use in nanolang

```nano
import "onnx"

fn main() -> int {
    (println "Loading ONNX model...")
    let model: int = (onnx_load_model "simple_model.onnx")
    
    if (< model 0) {
        (println "Failed to load model")
        return 1
    } else {
        (println "Model loaded successfully!")
        (onnx_free_model model)
        return 0
    }
}

shadow main {
    # Skipped - uses extern functions
}
```

## API Reference

### Core Functions

#### `onnx_load_model(path: string) -> int`
Load an ONNX model from file.
- **Returns**: Model handle (>= 0) on success, -1 on failure
- **Example**: `let model: int = (onnx_load_model "model.onnx")`

#### `onnx_free_model(model: int) -> void`
Free a loaded model and release resources.
- **Example**: `(onnx_free_model model)`

#### `onnx_run_inference(...) -> int`
Run inference on a model (low-level interface).
- **Returns**: 0 on success, -1 on failure

#### `onnx_get_input_shape(model: int, shape_out: int, max_dims: int) -> int`
Get model input shape information.
- **Returns**: Number of dimensions

#### `onnx_get_output_shape(model: int, shape_out: int, max_dims: int) -> int`
Get model output shape information.
- **Returns**: Number of dimensions

## Limitations

### Current Version
- Single input, single output models only
- Float32 tensors only (double/float in nanolang)
- Fixed shapes (no dynamic batching yet)
- CPU inference only

### Future Enhancements
- [ ] Multi-input/output models
- [ ] Dynamic batching
- [ ] GPU acceleration support
- [ ] Quantized models (int8)
- [ ] Model metadata queries

## Model Sources

### Pre-trained Models

1. **ONNX Model Zoo**: https://github.com/onnx/models
   - Image classification (ResNet, MobileNet, VGG)
   - Object detection (YOLO, SSD)
   - Segmentation models
   - NLP models (BERT, GPT)

2. **Hugging Face**: https://huggingface.co/models
   - Filter by "ONNX" tag
   - Many NLP and vision models

3. **PyTorch Hub**: Export with `torch.onnx.export()`

4. **TensorFlow**: Convert with `tf2onnx`

### Model Conversion

**From PyTorch:**
```python
torch.onnx.export(model, dummy_input, "model.onnx")
```

**From TensorFlow:**
```bash
pip install tf2onnx
python -m tf2onnx.convert --saved-model model_dir --output model.onnx
```

**From scikit-learn:**
```python
from skl2onnx import convert_sklearn
onnx_model = convert_sklearn(sklearn_model, initial_types=[...])
```

## Examples

See the `examples/` directory:
- `examples/40_onnx_simple.nano` - Basic model loading
- `examples/41_onnx_inference.nano` - Simple inference example

## Troubleshooting

### "Failed to load model"
- Check file path is correct
- Verify model is valid ONNX format: `python -m onnxruntime.tools.check_onnx_model model.onnx`
- Ensure model has 1 input and 1 output

### "Failed to create session"
- Check ONNX Runtime is installed: `pkg-config --modversion onnxruntime`
- Try reinstalling: `brew reinstall onnxruntime`

### Compilation Errors
```bash
# Check if ONNX Runtime headers are available
ls /opt/homebrew/include/onnxruntime_c_api.h

# Check if library is available
ls /opt/homebrew/lib/libonnxruntime*
```

## Performance Tips

1. **Use smaller models**: MobileNet instead of ResNet152
2. **Quantize models**: Convert to int8 for faster inference
3. **Batch processing**: Process multiple inputs together (future feature)
4. **Cache models**: Load once, infer many times

## Related Documentation

- [ONNX Runtime Docs](https://onnxruntime.ai/docs/)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)

## License

This module wraps ONNX Runtime, which is licensed under the MIT License.

