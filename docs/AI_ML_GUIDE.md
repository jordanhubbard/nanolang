# I Support AI and Machine Learning

**Version:** 1.0  
**Date:** February 20, 2026  
**Status:** Available

---

## Overview

I support AI and machine learning through my **ONNX Runtime module**. I enable CPU-based neural network inference without requiring a GPU.

**What I offer:**

- I run pre-trained neural networks.
- I am CPU-only. I do not require CUDA or a GPU.
- I support PyTorch, TensorFlow, and scikit-learn models through the ONNX format.
- I handle image classification, NLP, and object detection.
- I am cross-platform. I run on macOS and Linux.

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

### 2. Import My ONNX Module

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

I load an ONNX model from a file.

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

I free model resources.

**Parameters:**
- `model`: Model handle from `onnx_load_model`

**Example:**
```nano
(onnx_free_model model)
```

#### `onnx_run_inference(...) -> int`

I run inference on a model through this low-level interface.

**Note:** My full inference support requires runtime array-to-pointer conversion. I have planned this for a future update.

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

This is the official repository of pre-trained models:
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

# Use in my environment
let model: int = (onnx_load_model "resnet50-v2-7.onnx")
```

---

## Use Cases

### Image Classification

**Problem:** Identify what is in an image.

**Models:** ResNet, MobileNet, VGG, EfficientNet.

**Input:** RGB image (e.g., 224, 224, 3).

**Output:** Class probabilities.

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

**Problem:** Find and locate objects in an image.

**Models:** YOLO, SSD, Faster R-CNN.

**Input:** RGB image.

**Output:** Bounding boxes, class labels, and confidence scores.

### Sentiment Analysis

**Problem:** Determine if text is positive, negative, or neutral.

**Models:** BERT, DistilBERT, RoBERTa.

**Input:** Tokenized text.

**Output:** Sentiment class.

### Time Series Forecasting

**Problem:** Predict future values from historical data.

**Models:** LSTM, GRU, Temporal Convolutional Networks.

**Input:** Historical values.

**Output:** Future predictions.

---

## Performance Optimization

### 1. Use Smaller Models

**Problem:** Large models are slow.

**Solution:** I recommend using lightweight alternatives.

| Task | Heavy Model | Light Model | Speedup |
|------|-------------|-------------|---------|
| Image Classification | ResNet152 | MobileNetV2 | 10x faster |
| Object Detection | Mask R-CNN | YOLOv4-tiny | 5x faster |
| NLP | BERT-large | DistilBERT | 2x faster |

### 2. Quantize Models

**Problem:** Float32 models are large and slow.

**Solution:** Convert them to int8.

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
- File size is 4x smaller.
- Inference is 2 to 4 times faster.
- Accuracy loss is minimal.

### 3. Optimize Input Size

**Problem:** Large inputs slow down processing.

**Solution:** Resize images to the minimum required size.

```python
# Instead of 1024x1024, use 224x224 if model supports it
image = resize(image, (224, 224))
```

### 4. Batch Processing

**Problem:** Processing one item at a time is inefficient.

**Solution:** I will support processing multiple items together in a future update.

---

## My Current Limitations

### Version 1.0 Limitations

1. **Single Input/Output Only**
   - I require models to have exactly 1 input and 1 output.
   - I do not yet support multi-input or multi-output models.

2. **Fixed Shapes**
   - I have not implemented dynamic batching.
   - You must use a batch size of 1.

3. **Float32 Only**
   - I only support float tensors.
   - I have not yet tested int8 quantized models.

4. **CPU Only**
   - I do not offer GPU acceleration yet.
   - All inference runs on the CPU.

5. **Array-to-Pointer Conversion**
   - I am currently working on the runtime support for full inference.
   - I am limited to model loading and unloading for now.

### My Planned Features

- [ ] Multi-input and multi-output models
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
- `tiny_classifier.onnx` - Tiny CNN for 32x32 images

### Example 3: Image Classifier (Conceptual)

```nano
import "modules/onnx/onnx.nano"

fn classify_image(model_path: string, image_path: string) -> int {
    # I load the model
    let model: int = (onnx_load_model model_path)
    if (< model 0) {
        return -1
    } else {
        # I will prepare the image when full support is available
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
- ONNX Runtime is not installed.
- The model file does not exist.
- The model has multiple inputs or outputs.
- The model is corrupted.

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

**Problem:** Inference is too slow.

**Solutions:**
1. Use a smaller model.
2. Quantize to int8.
3. Reduce the input size.
4. I will add optimizations for ONNX Runtime in a future release.

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
2. **Run my example:** `examples/40_onnx_simple.nano`
3. **Create test models:** `examples/models/create_test_model.py`
4. **Load and explore models**

### For Intermediate Users

1. **Export PyTorch or TensorFlow models to ONNX**
2. **Download pre-trained models from the ONNX Model Zoo**
3. **Experiment with different model architectures**
4. **Optimize models for inference**

### For Advanced Users

1. **Quantize models to int8**
2. **Create custom model architectures**
3. **Benchmark inference performance**
4. **Contribute enhancements to my ONNX module**

---

## Comparison: Me vs Other Languages

| Feature | Me + ONNX | Python + PyTorch | C++ + LibTorch |
|---------|-----------------|------------------|----------------|
| Easy to learn | Very easy | Easy | Complex |
| Startup time | Fast | Slow | Fast |
| Memory usage | Low | High | Low |
| Deployment | Single binary | Dependencies | Single binary |
| Training | Not supported | Full support | Full support |
| Inference | Supported | Supported | Supported |
| Model formats | ONNX | PyTorch/ONNX | PyTorch/ONNX |

**Use me when:**
- You need fast, lightweight inference.
- You want single-binary deployment.
- You are building embedded or edge applications.
- You do not need training.

**Use Python when:**
- You need to train models.
- You need the full ML ecosystem.
- Development speed is your priority.

---

## Contributing

If you want to improve my AI capabilities:

1. **Add tensor types** - I need first-class tensor support.
2. **GPU support** - I need CUDA or Metal acceleration.
3. **More model formats** - I could support TensorFlow Lite or CoreML.
4. **Training support** - I need backpropagation and optimizers.
5. **Quantization** - I need int8 or int16 inference.

See my `planning/` directory for roadmap and design docs.

---

## License

ONNX Runtime is licensed under the MIT License.

My ONNX module is part of me and follows my license.



