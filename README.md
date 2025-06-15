# 🐱🐶 Cat vs Dog Image Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-d00000?style=for-the-badge&logo=keras&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-00c851?style=for-the-badge)

**A deep learning CNN model for binary image classification using TensorFlow and Keras**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Architecture](#-architecture)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Dataset Structure](#-dataset-structure)
- [Usage](#-usage)
- [Results](#-results)
- [Technical Details](#-technical-details)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Overview

This project implements a **Convolutional Neural Network (CNN)** for classifying images of cats and dogs with high accuracy. The model leverages both **Transfer Learning** with MobileNetV2 and a custom CNN architecture, featuring advanced data augmentation techniques and robust training strategies.

### 🎯 Key Highlights

- **Dual Architecture Support**: Transfer Learning (MobileNetV2) + Custom CNN
- **Advanced Data Augmentation** for improved generalization
- **Smart Training Strategies**: Early stopping, learning rate reduction
- **Production Ready**: Complete preprocessing and prediction pipeline
- **Comprehensive Visualization** of training metrics

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Transfer Learning** | Pre-trained MobileNetV2 for superior performance |
| 🔄 **Data Augmentation** | Horizontal flip, zoom, width/height shifts |
| 📊 **Smart Callbacks** | Early stopping and adaptive learning rate |
| 🎨 **Visualization** | Training history plots and performance metrics |
| ⚡ **Fast Inference** | Optimized for quick predictions |
| 🛡️ **Regularization** | Dropout layers prevent overfitting |

---

## 🏗️ Model Architecture

### Transfer Learning Architecture (Default)
```
MobileNetV2 (ImageNet weights, frozen)
           ↓
   GlobalAveragePooling2D
           ↓
      Dropout(0.3)
           ↓
    Dense(128, ReLU)
           ↓
      Dropout(0.5)
           ↓
   Dense(1, Sigmoid)
```

### Custom CNN Architecture (Alternative)
```
Input(160×160×3)
       ↓
Conv2D(16) + BatchNorm + MaxPool + Dropout(0.25)
       ↓
Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
       ↓
Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
       ↓
GlobalAveragePooling2D
       ↓
Dense(32, ReLU) + Dropout(0.5)
       ↓
Dense(1, Sigmoid)
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/Wayn-Git/CatvsDog.git
cd CatvsDog

# Install dependencies
pip install tensorflow matplotlib seaborn numpy
```

### Dependencies
```bash
tensorflow>=2.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
numpy>=1.21.0
```

---

## 📁 Dataset Structure

```
CatvsDog/
├── Data/
│   ├── train/
│   │   ├── cats/          # Training cat images
│   │   └── dogs/          # Training dog images
│   └── test/
│       ├── cats/          # Test cat images
│       └── dogs/          # Test dog images
├── Model/
│   └── cat_dog_model.keras
├── Notebook/
│   ├── CatvsDog/
│   ├── (Backup).ipynb
│   └── catDog(Col...).ipynb
├── PythonScript/
│   ├── catdog.py          # Main training script
│   ├── catDog.ipynb
│   ├── README.md
│   └── requirements.txt
└── README.md
```

---

## 🚀 Usage

### Training the Model

```python
# Set transfer learning mode
TRANSFER_LEARNING = True  # Use MobileNetV2 (recommended)
# TRANSFER_LEARNING = False  # Use custom CNN

# Run training
python PythonScript/catdog.py
```

### Making Predictions

```python
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Load trained model
model = load_model('Model/cat_dog_model.keras')

# Load and preprocess image
img_path = "your_image.jpg"
img = image.load_img(img_path, target_size=(160, 160))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
result = "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"
print(f"Predicted: {result}")
print(f"Confidence: {prediction[0][0]:.4f}")
```

---

## 📈 Results

### Model Performance

| Architecture | Test Accuracy | Training Time | Model Size |
|--------------|---------------|---------------|------------|
| **Transfer Learning (MobileNetV2)** | ~95%+ | ~15-20 epochs | ~9MB |
| **Custom CNN** | ~85-90% | ~25-30 epochs | ~3MB |

### Training Features

- **Early Stopping**: Prevents overfitting with patience=5
- **Learning Rate Reduction**: Adaptive LR with factor=0.2, patience=3
- **Data Augmentation**: Improves generalization significantly
- **Batch Processing**: Efficient training with batch_size=64

---

## 🔧 Technical Details

### Data Preprocessing

```python
# Training augmentation
trainGen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Test preprocessing
testGen = ImageDataGenerator(rescale=1./255)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Input Size** | 160×160×3 |
| **Batch Size** | 64 |
| **Optimizer** | Adam |
| **Loss Function** | Binary Crossentropy |
| **Max Epochs** | 30 |
| **Early Stopping Patience** | 5 |
| **LR Reduction Factor** | 0.2 |

### Key Features

- **Robust Architecture**: Both transfer learning and custom CNN options
- **Smart Callbacks**: Early stopping and learning rate scheduling
- **Data Augmentation**: Comprehensive image transformations
- **Regularization**: Strategic dropout placement
- **Efficient Training**: Optimized batch processing

---

## 📊 Visualization

The model automatically generates training visualizations:

- **Accuracy Curves**: Training vs Validation accuracy over epochs
- **Loss Curves**: Training vs Validation loss progression
- **Performance Metrics**: Detailed evaluation results

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/improvement`)
3. **Commit** your changes (`git commit -am 'Add new feature'`)
4. **Push** to the branch (`git push origin feature/improvement`)
5. **Create** a Pull Request

### Areas for Contribution
- Model architecture improvements
- Additional data augmentation techniques
- Web interface development
- Mobile deployment optimization
- Documentation enhancements

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **TensorFlow/Keras** team for the excellent deep learning framework
- **MobileNetV2** architecture for efficient transfer learning
- **ImageNet** dataset for pre-trained weights
- Open source community for inspiration and support

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ and Deep Learning

[Report Bug](https://github.com/Wayn-Git/CatvsDog/issues) • [Request Feature](https://github.com/Wayn-Git/CatvsDog/issues) • [Documentation](https://github.com/Wayn-Git/CatvsDog/wiki)

</div>
