# ONNX Test Models

This directory contains scripts to create test ONNX models for nanolang AI/ML examples.

## Quick Start

### 1. Install PyTorch

```bash
pip install torch onnx
```

### 2. Create Test Models

```bash
python create_test_model.py
```

This creates:
- `simple_model.onnx` - Basic feedforward neural network (10 → 20 → 5)
- `tiny_classifier.onnx` - Small CNN for 32×32 grayscale images

### 3. Run Examples

```bash
# From project root
./bin/nanoc examples/40_onnx_simple.nano -o onnx_simple
./onnx_simple

./bin/nanoc examples/41_onnx_inference.nano -o onnx_inference
./onnx_inference

./bin/nanoc examples/42_onnx_classifier.nano -o onnx_classifier
./onnx_classifier
```

## Model Details

### simple_model.onnx

- **Input**: `[batch, 10]` - 10 float values
- **Output**: `[batch, 5]` - 5 float values
- **Layers**: Linear(10→20) + ReLU + Linear(20→5)
- **Use case**: Basic inference testing

### tiny_classifier.onnx

- **Input**: `[batch, 1, 32, 32]` - Grayscale 32×32 images
- **Output**: `[batch, 10]` - 10 class logits
- **Architecture**: CNN with 2 conv layers + 2 FC layers
- **Use case**: Image classification testing

## Pre-trained Models

For real applications, download pre-trained models:

### ONNX Model Zoo
```bash
# ResNet50 (image classification)
wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx

# MobileNetV2 (lightweight classifier)
wget https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx

# YOLO (object detection)
wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov4/model/yolov4.onnx
```

### Hugging Face

Visit https://huggingface.co/models and filter by "ONNX" tag for NLP models.

## Creating Your Own Models

### From PyTorch

```python
import torch
import torch.nn as nn

# Define model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

# Export
model = MyModel()
model.eval()
dummy_input = torch.randn(1, 10)
torch.onnx.export(model, dummy_input, "my_model.onnx")
```

### From TensorFlow

```bash
pip install tf2onnx
python -m tf2onnx.convert --saved-model model_dir --output model.onnx
```

### From scikit-learn

```python
from skl2onnx import convert_sklearn
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Convert
onnx_model = convert_sklearn(model, initial_types=[...])
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

## Verification

Verify models are valid:

```bash
pip install onnx
python -c "import onnx; onnx.checker.check_model('simple_model.onnx')"
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

Install PyTorch:
```bash
pip install torch
```

### "Model has multiple inputs/outputs"

Currently, nanolang ONNX module only supports single-input, single-output models. Modify your model architecture or wait for multi-I/O support.

### Model too large

For development, use smaller models:
- MobileNetV2 instead of ResNet152
- DistilBERT instead of BERT-large
- YOLOv4-tiny instead of YOLOv4

## Related Documentation

- [AI_ML_GUIDE.md](../../docs/AI_ML_GUIDE.md) - Complete AI/ML guide
- [ONNX Module README](../../modules/onnx/README.md) - ONNX module documentation
- [ONNX Runtime Docs](https://onnxruntime.ai/docs/) - Official ONNX Runtime documentation

