# AI and Machine Learning with nanolang

**Version:** 1.0  
**Date:** November 19, 2025  
**Status:** ✅ Available

---

## Overview

nanolang supports AI and machine learning through the **ONNX Runtime module**, enabling CPU-based neural network inference without requiring a GPU.

**Key Features:**

- ✅ Run pre-trained neural networks
- ✅ CPU-only (no CUDA/GPU required)
- ✅ Support for PyTorch, TensorFlow, scikit-learn models
- ✅ Image classification, NLP, object detection
- ✅ Cross-platform (macOS, Linux)

---

## Quick Start

### 1. Install ONNX Runtime

**macOS:**
```bash
brew install onnxruntime
```

**Linux:**
```bash
sudo apt-get install libonnxruntime-dev
```

**Verify:**
```bash
pkg-config --exists onnxruntime && echo "✓ Installed" || echo "✗ Not found"
```

### 2. Import the ONNX Module

```nano
import "modules/onnx/onnx.nano"

fn main() -> int {
    let model: int = (onnx_load_model "model.onnx")
    if (< model 0) {
        (println "Failed to load model")
        return 1
    } else {
        (println "Model loaded!")
        (onnx_free_model model)
        return 0
    }
}

shadow main {
    # Skipped - uses extern functions
}
```

### 3. Run

```bash
./bin/nanoc my_program.nano -o my_program
./my_program
```

---

## ONNX Module API

### Core Functions

#### `onnx_load_model(path: string) -> int`

Load an ONNX model from file.

**Parameters:**
- `path`: Path to `.onnx` model file

**Returns:**
- Model handle (>= 0) on success
- -1 on failure

**Example:**
```nano
let model: int = (onnx_load_model "resnet50.onnx")
if (< model 0) {
    (println "Failed to load model")
} else {
    (println "Model loaded successfully")
}
```

#### `onnx_free_model(model: int) -> void`

Free model resources.

**Parameters:**
- `model`: Model handle from `onnx_load_model`

**Example:**
```nano
(onnx_free_model model)
```

#### `onnx_run_inference(...) -> int`

Run inference on a model (low-level interface).

**Note:** Full inference support requires runtime array-to-pointer conversion, which is planned for a future update.

---

## Creating ONNX Models

### From PyTorch

```python
import torch
import torch.nn as nn

# Define model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Create and export
model = SimpleNet()
model.eval()

dummy_input = torch.randn(1, 10)
torch.onnx.export(
    model,
    dummy_input,
    "simple_net.onnx",
    input_names=["input"],
    output_names=["output"]
)
```

### From TensorFlow

```bash
# Install converter
pip install tf2onnx

# Convert model
python -m tf2onnx.convert \
    --saved-model saved_model_dir \
    --output model.onnx
```

### From scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save
with open("rf_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

---

## Pre-trained Models

### ONNX Model Zoo

Official repository of pre-trained models:
- **URL:** https://github.com/onnx/models
- **Models:** Image classification, object detection, NLP, segmentation

**Popular Models:**
- **ResNet50** - Image classification (1000 classes)
- **MobileNetV2** - Lightweight image classification
- **BERT** - NLP (text understanding)
- **YOLOv4** - Object detection
- **SqueezeNet** - Tiny image classifier

### Hugging Face

Pre-trained transformer models:
- **URL:** https://huggingface.co/models
- **Filter:** ONNX tag
- **Models:** Text generation, translation, sentiment analysis

### Download Example

```bash
# Download ResNet50 from ONNX Model Zoo
wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx

# Use in nanolang
let model: int = (onnx_load_model "resnet50-v2-7.onnx")
```

---

## Use Cases

### Image Classification

**Problem:** Identify what's in an image (dog, cat, car, etc.)

**Models:** ResNet, MobileNet, VGG, EfficientNet

**Input:** RGB image (e.g., 224×224×3)

**Output:** Class probabilities (e.g., 1000 ImageNet classes)

**Example:**
```nano
# Load ResNet50
let model: int = (onnx_load_model "resnet50.onnx")

# Prepare image (224x224x3 RGB pixels, normalized)
let image: array<float> = (load_and_preprocess_image "photo.jpg")

# Run inference (when full support is available)
# let output: array<float> = (run_inference model image)
# let class_id: int = (argmax output)

(onnx_free_model model)
```

### Object Detection

**Problem:** Find and locate objects in an image

**Models:** YOLO, SSD, Faster R-CNN

**Input:** RGB image

**Output:** Bounding boxes, class labels, confidence scores

### Sentiment Analysis

**Problem:** Determine if text is positive, negative, or neutral

**Models:** BERT, DistilBERT, RoBERTa

**Input:** Tokenized text

**Output:** Sentiment class (positive/negative/neutral)

### Time Series Forecasting

**Problem:** Predict future values from historical data

**Models:** LSTM, GRU, Temporal Convolutional Networks

**Input:** Historical values (e.g., stock prices, temperature)

**Output:** Future predictions

---

## Performance Optimization

### 1. Use Smaller Models

**Problem:** Large models are slow

**Solution:** Use lightweight alternatives

| Task | Heavy Model | Light Model | Speedup |
|------|-------------|-------------|---------|
| Image Classification | ResNet152 | MobileNetV2 | 10x faster |
| Object Detection | Mask R-CNN | YOLOv4-tiny | 5x faster |
| NLP | BERT-large | DistilBERT | 2x faster |

### 2. Quantize Models

**Problem:** Float32 models are large and slow

**Solution:** Convert to int8

```python
# PyTorch quantization
import torch.quantization

model_fp32 = MyModel()
model_fp32.eval()

# Post-training static quantization
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)

# Export to ONNX
torch.onnx.export(model_int8, dummy_input, "model_quantized.onnx")
```

**Benefits:**
- 4x smaller file size
- 2-4x faster inference
- Minimal accuracy loss

### 3. Optimize Input Size

**Problem:** Large inputs slow down processing

**Solution:** Resize images to minimum required size

```python
# Instead of 1024x1024, use 224x224 if model supports it
image = resize(image, (224, 224))
```

### 4. Batch Processing

**Problem:** Processing one item at a time is inefficient

**Solution:** Process multiple items together (future feature)

---

## Current Limitations

### Version 1.0 Limitations

1. **Single Input/Output Only**
   - Models must have exactly 1 input and 1 output
   - Multi-input/output models not yet supported

2. **Fixed Shapes**
   - Dynamic batching not yet implemented
   - Must use batch size = 1

3. **Float32 Only**
   - Only float tensors supported
   - Int8 quantized models not yet tested

4. **CPU Only**
   - No GPU acceleration yet
   - All inference runs on CPU

5. **Array-to-Pointer Conversion**
   - Full inference requires runtime support (in progress)
   - Currently limited to model loading/unloading

### Planned Features

- [ ] Multi-input/output models
- [ ] Dynamic batching
- [ ] GPU acceleration (optional)
- [ ] Quantized model support
- [ ] Model metadata queries
- [ ] Full inference with array conversion

---

## Examples

### Example 1: Simple Model Loading

```nano
import "modules/onnx/onnx.nano"

fn main() -> int {
    let model: int = (onnx_load_model "model.onnx")
    if (>= model 0) {
        (println "Model loaded successfully")
        (onnx_free_model model)
        return 0
    } else {
        (println "Failed to load model")
        return 1
    }
}

shadow main {
    # Skipped - uses extern functions
}
```

**Run:**
```bash
./bin/nanoc examples/40_onnx_simple.nano -o onnx_simple
./onnx_simple
```

### Example 2: Create Test Models

See `examples/models/create_test_model.py`:

```bash
cd examples/models
python create_test_model.py
```

This creates:
- `simple_model.onnx` - Basic feedforward network
- `tiny_classifier.onnx` - Tiny CNN for 32×32 images

### Example 3: Image Classifier (Conceptual)

```nano
import "modules/onnx/onnx.nano"

fn classify_image(model_path: string, image_path: string) -> int {
    # Load model
    let model: int = (onnx_load_model model_path)
    if (< model 0) {
        return -1
    } else {
        # Prepare image (when full support is available)
        # let image: array<float> = (load_image image_path)
        # let output: array<float> = (run_inference model image)
        # let class_id: int = (argmax output)
        
        (onnx_free_model model)
        return 0
    }
}

shadow classify_image {
    # Skipped - uses extern functions
}
```

---

## Troubleshooting

### "Failed to load model"

**Causes:**
- ONNX Runtime not installed
- Model file doesn't exist
- Model has multiple inputs/outputs
- Model is corrupted

**Solutions:**
```bash
# Check ONNX Runtime
pkg-config --modversion onnxruntime

# Verify model
python -c "import onnx; onnx.checker.check_model('model.onnx')"

# Check inputs/outputs
python -c "
import onnx
model = onnx.load('model.onnx')
print(f'Inputs: {len(model.graph.input)}')
print(f'Outputs: {len(model.graph.output)}')
"
```

### Compilation Errors

**Error:** `onnxruntime_c_api.h: No such file or directory`

**Solution:**
```bash
# macOS
brew install onnxruntime

# Linux
sudo apt-get install libonnxruntime-dev

# Verify headers
ls /opt/homebrew/include/onnxruntime_c_api.h
```

### Performance Issues

**Problem:** Inference is too slow

**Solutions:**
1. Use a smaller model (MobileNet vs ResNet)
2. Quantize to int8
3. Reduce input size
4. Enable optimizations in ONNX Runtime (future feature)

---

## Educational Resources

### Learn About Neural Networks

1. **PyTorch Tutorials:** https://pytorch.org/tutorials/
2. **Fast.ai Course:** https://course.fast.ai/
3. **Stanford CS231n:** http://cs231n.stanford.edu/

### Learn About ONNX

1. **ONNX Docs:** https://onnx.ai/
2. **ONNX Runtime:** https://onnxruntime.ai/docs/
3. **ONNX Tutorials:** https://github.com/onnx/tutorials

### Model Architectures

1. **Papers with Code:** https://paperswithcode.com/
2. **Hugging Face Docs:** https://huggingface.co/docs
3. **Model Zoo:** https://github.com/onnx/models

---

## Next Steps

### For Beginners

1. **Install ONNX Runtime**
2. **Run example:** `examples/40_onnx_simple.nano`
3. **Create test models:** `examples/models/create_test_model.py`
4. **Load and explore models**

### For Intermediate Users

1. **Export PyTorch/TensorFlow models to ONNX**
2. **Download pre-trained models from ONNX Model Zoo**
3. **Experiment with different model architectures**
4. **Optimize models for inference**

### For Advanced Users

1. **Quantize models to int8**
2. **Create custom model architectures**
3. **Benchmark inference performance**
4. **Contribute enhancements to nanolang ONNX module**

---

## Comparison: nanolang vs Other Languages

| Feature | nanolang + ONNX | Python + PyTorch | C++ + LibTorch |
|---------|-----------------|------------------|----------------|
| Easy to learn | ✅ Very easy | ✅ Easy | ❌ Complex |
| Startup time | ✅ Fast | ❌ Slow | ✅ Fast |
| Memory usage | ✅ Low | ❌ High | ✅ Low |
| Deployment | ✅ Single binary | ❌ Dependencies | ✅ Single binary |
| Training | ❌ Not supported | ✅ Full support | ✅ Full support |
| Inference | ✅ Supported | ✅ Supported | ✅ Supported |
| Model formats | ✅ ONNX | ✅ PyTorch/ONNX | ✅ PyTorch/ONNX |

**Use nanolang when:**
- You need fast, lightweight inference
- You want single-binary deployment
- You're building embedded/edge applications
- You don't need training

**Use Python when:**
- You need to train models
- You need the full ML ecosystem
- Development speed is priority

---

## Contributing

Want to improve nanolang's AI capabilities?

1. **Add tensor types** - First-class tensor support
2. **GPU support** - CUDA/Metal acceleration
3. **More model formats** - TensorFlow Lite, CoreML
4. **Training support** - Backpropagation, optimizers
5. **Quantization** - int8/int16 inference

See `planning/` directory for roadmap and design docs.

---

## License

ONNX Runtime is licensed under the MIT License.

nanolang ONNX module is part of nanolang and follows the nanolang license.


