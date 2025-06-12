# 🐱🐶 Cats vs Dogs Image Classifier

> A deep learning CNN model built with TensorFlow/Keras for binary image classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow.svg)](#status)

## ⚠️ Project Status

**🚧 This project is currently under development and may contain bugs or incomplete features.**

- Currently debugging and optimizing the model
- Performance metrics are being fine-tuned
- Some features may not work as expected
- Active development in progress

## 🚀 Overview

This project implements a **Convolutional Neural Network (CNN)** to classify images of cats and dogs with high accuracy. The model uses advanced data augmentation techniques and a well-structured architecture to achieve robust performance on the classic computer vision problem.

### ✨ Planned Features

- **Deep CNN Architecture** with 4 convolutional layers
- **Data Augmentation** for improved generalization
- **Dropout Regularization** to prevent overfitting
- **Real-time Prediction** capabilities *(in development)*
- **Comprehensive Visualization** of training metrics

## 🏗️ Model Architecture

```
Input (150x150x3)
    ↓
Conv2D(32) + ReLU → MaxPool2D
    ↓
Conv2D(64) + ReLU → MaxPool2D
    ↓
Conv2D(128) + ReLU → MaxPool2D
    ↓
Conv2D(128) + ReLU → MaxPool2D
    ↓
Flatten → Dropout(0.5)
    ↓
Dense(512) + ReLU
    ↓
Dense(1) + Sigmoid
```

## 📊 Data Processing

The model employs sophisticated data augmentation techniques:

- **Rescaling**: Normalized pixel values (0-1)
- **Horizontal Flip**: Random horizontal flipping
- **Rotation**: Up to 40° random rotation
- **Zoom**: 20% random zoom
- **Translation**: Width/height shifts up to 20%
- **Shear**: Shear transformation up to 20%

## 🛠️ Installation

### Prerequisites

```bash
pip install tensorflow>=2.8.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install numpy>=1.21.0
```

### Quick Setup

```bash
git clone https://github.com/yourusername/cats-dogs-classifier.git
cd cats-dogs-classifier
pip install -r requirements.txt
```

## 📁 Project Structure

```
cats-dogs-classifier/
├── Data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
├── models/
├── notebooks/
├── src/
│   └── classifier.py
├── requirements.txt
└── README.md
```

## 🚀 Usage

⚠️ **Development Notice**: Some functionality may be unstable during active development.

### Training the Model

```python
# Run the main training script (may require debugging)
python src/classifier.py
```

### Making Predictions *(Beta)*

```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess image
img_path = "path/to/your/image.jpg"
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction (accuracy may vary)
prediction = model.predict(img_array)
result = "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"
print(f"Predicted: {result}")
```

> **Note**: Prediction accuracy is being improved. Results may not be reliable yet.

## 📈 Current Results

### Model Performance *(Preliminary)*

| Metric | Status |
|--------|--------|
| **Test Accuracy** | In Progress 🔄 |
| **Training Time** | ~25 epochs |
| **Model Size** | ~12MB |

> **Note**: Performance metrics are being optimized and may vary during development.

### Known Issues & Debugging

- ⚠️ Model convergence being optimized
- ⚠️ Data preprocessing pipeline under review
- ⚠️ Validation accuracy fluctuations
- ⚠️ Memory usage optimization needed

### Training Visualization

The model generates training plots showing:
- **Accuracy curves** (Training vs Validation) *(may show instability)*
- **Loss curves** (Training vs Validation) *(debugging in progress)*
- **Performance metrics** over epochs

## 🔧 Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Input Size** | 150x150x3 |
| **Batch Size** | 32 |
| **Learning Rate** | 1e-4 |
| **Optimizer** | Adam |
| **Loss Function** | Binary Crossentropy |
| **Epochs** | 25 |
| **Dropout Rate** | 0.5 |

## 🎯 Key Techniques

- **Transfer Learning Ready**: Architecture can be adapted for transfer learning
- **Regularization**: Dropout layers prevent overfitting
- **Data Augmentation**: Increases dataset diversity artificially
- **Early Stopping**: Can be implemented for optimal training
- **Learning Rate Scheduling**: Adaptive learning rate with Adam optimizer

## 🔍 Model Insights

### Architecture Benefits
- **Progressive Feature Extraction**: Filters increase from 32→64→128→128
- **Spatial Reduction**: MaxPooling reduces spatial dimensions progressively
- **Non-linearity**: ReLU activation functions enable complex pattern learning
- **Regularization**: Dropout prevents overfitting on training data

### Performance Optimization
- **Batch Processing**: Efficient GPU utilization with batch size 32
- **Memory Efficiency**: Progressive downsampling reduces memory usage
- **Gradient Stability**: Adam optimizer with conservative learning rate

## 🔧 Current Development

### Active Tasks
- [ ] **Model Optimization**: Fine-tuning hyperparameters
- [ ] **Bug Fixes**: Resolving training instabilities
- [ ] **Data Pipeline**: Improving data loading efficiency
- [ ] **Validation**: Implementing proper train/val split
- [ ] **Documentation**: Adding comprehensive code comments

### Debug Log
- Investigating overfitting issues
- Optimizing learning rate scheduling
- Fixing data augmentation parameters
- Resolving memory leaks during training

## 🚀 Future Enhancements

- [ ] **Transfer Learning** with pre-trained models (VGG16, ResNet50)
- [ ] **Advanced Data Augmentation** techniques
- [ ] **Ensemble Methods** for improved accuracy
- [ ] **Web Interface** for easy image upload and prediction
- [ ] **Mobile Deployment** with TensorFlow Lite
- [ ] **Real-time Video Classification**
- [ ] **Model Interpretability** with Grad-CAM

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

*I welcome contributions!** This project is actively being developed and debugged.

### How to Help
- 🐛 **Bug Reports**: Found an issue? Please create a detailed issue report
- 🔧 **Bug Fixes**: Submit PRs for any bugs you can fix
- 💡 **Suggestions**: Ideas for improvement are always welcome
- 📖 **Documentation**: Help improve code documentation
- 🧪 **Testing**: Help with testing different configurations

### Getting Started
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Test your changes thoroughly
4. Submit a pull request with detailed description

> **Note**: Please test your contributions as the codebase is still being stabilized.


---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**

Made with ❤️ and TensorFlow

</div>
