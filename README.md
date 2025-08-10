# 🌱 Plant Doctor - AI-Powered Plant Disease Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent web application that uses deep learning to detect and classify plant diseases from leaf images. Built with TensorFlow and Streamlit, this tool helps farmers and gardeners identify plant health issues quickly and accurately.

## 🌟 Features

- **🔍 Real-time Disease Detection**: Upload plant leaf images and get instant disease predictions
- **🌿 Multi-Plant Support**: Detects diseases in 14 different plant types including:
  - Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach
  - Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
- **📊 38 Disease Categories**: Covers both healthy plants and various disease conditions
- **🎨 Modern UI**: Beautiful, responsive interface with light/dark theme support
- **📱 Mobile-Friendly**: Works seamlessly on desktop and mobile devices
- **⚡ Fast Processing**: Optimized model for quick predictions
- **📈 High Accuracy**: Trained on extensive dataset with 95%+ validation accuracy

## 🏗️ Project Structure

```
Plant_Disease_Prediction/
├── app.py                          # Main Streamlit application
├── trained_model.keras             # Pre-trained deep learning model
├── training_hist.json              # Training history and metrics
├── requirements.txt                # Python dependencies
├── Train_plant_disease.ipynb       # Model training notebook
├── Test_Plant_Disease.ipynb        # Model testing notebook
├── Plant_Disease_Dataset/          # Dataset directory
│   ├── train/                      # Training images
│   ├── valid/                      # Validation images
│   └── test/                       # Test images
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Plant_Disease_Prediction.git
   cd Plant_Disease_Prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, manually navigate to the URL

## 📖 How to Use

1. **Upload an Image**
   - Click on the file uploader in the "🔍 Detect Disease" tab
   - Select a clear image of a plant leaf (JPG, JPEG, or PNG format)
   - Ensure the leaf is well-lit and clearly visible

2. **Analyze the Plant**
   - Click the "🔬 Analyze Plant" button
   - Wait for the AI to process your image (usually takes 2-5 seconds)

3. **View Results**
   - The app will display:
     - Plant type detected
     - Disease status (Healthy or Disease detected)
     - Specific disease name (if applicable)
     - Confidence level of the prediction
     - Description of the detected condition

## 🧠 Model Information

### Architecture
- **Framework**: TensorFlow 2.x
- **Model Type**: Convolutional Neural Network (CNN)
- **Input Size**: 128x128 pixels (RGB)
- **Output**: 38 classes (14 plants × various disease states)

### Performance Metrics
- **Training Accuracy**: 98.15%
- **Validation Accuracy**: 95.99%
- **Training Loss**: 0.055
- **Validation Loss**: 0.143

### Dataset
- **Source**: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Training Images**: 70,295
- **Validation Images**: 17,572
- **Classes**: 38 (including healthy plants)

## 🛠️ Development

### Training Your Own Model

1. **Prepare the dataset**
   - Download the dataset from Kaggle
   - Extract to `Plant_Disease_Dataset/` directory
   - Ensure proper folder structure with train/valid/test splits

2. **Run the training notebook**
   ```bash
   jupyter notebook Train_plant_disease.ipynb
   ```

3. **Test the model**
   ```bash
   jupyter notebook Test_Plant_Disease.ipynb
   ```

### Customizing the Application

- **Add new plant types**: Update the `DISEASE_CLASSES` dictionary in `app.py`
- **Modify UI**: Edit the CSS styles in the main function
- **Change model**: Replace `trained_model.keras` with your custom model

## 📋 Supported Plants and Diseases

| Plant | Diseases Detected |
|-------|------------------|
| **Apple** | Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn** | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Black Measles, Leaf Blight, Healthy |
| **Orange** | Citrus Greening |
| **Peach** | Bacterial Spot, Healthy |
| **Bell Pepper** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

## 🔧 Troubleshooting

### Common Issues

1. **Model loading error**
   - Ensure `trained_model.keras` is in the project root directory
   - Check if the file is corrupted (try re-downloading)

2. **Import errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Memory issues**
   - Close other applications to free up RAM
   - Consider using a smaller batch size in the model

4. **Slow predictions**
   - Ensure you have a decent GPU for faster processing
   - Close unnecessary browser tabs

### Performance Tips

- Use clear, well-lit images for better accuracy
- Ensure the leaf is centered and takes up most of the image
- Avoid shadows or reflections on the leaf surface
- Use images with resolution of at least 128x128 pixels
