
  
# Lung Cancer Detection using Deep Learning

This project implements a Convolutional Neural Network (CNN) based deep learning model to detect **lung cancer** from chest X-ray images. The model aims to assist medical professionals in early diagnosis of lung cancer using automated image classification.

# 📌 Project Objective

To develop an AI model that can classify chest X-ray images as either:

- **Lung Cancer Positive**
- **Lung Cancer Negative**

By leveraging deep learning techniques, this tool can potentially improve diagnosis speed and accuracy in clinical environments.



# 🧾 Dataset

The model was trained using a dataset of labeled chest X-ray images categorized into:
- `Cancer`
- `Non-Cancer`

> 📁 **Dataset should be organized as:**

LungCancerDataset/
├── Cancer/
│ ├── img1.jpg
│ ├── img2.jpg
├── Non-Cancer/
│ ├── img1.jpg
│ ├── img2.jpg

Copy
Edit

Alternatively, a CSV-based format can be used with file paths and labels.

---

## Model Architecture

- **Base Model**: Custom CNN / EfficientNet (based on notebook)
- **Input Shape**: 224x224x3
- **Layers**:
  - Convolutional Layers
  - Batch Normalization
  - MaxPooling
  - Dense Layers with Dropout
- **Output**: Sigmoid (binary classification)

---

## 🧪 Tools & Libraries

| Tool              | Purpose                     |
|------------------|-----------------------------|
| Python (Colab)    | Development environment     |
| TensorFlow/Keras | Deep learning framework     |
| OpenCV/PIL       | Image processing            |
| Matplotlib       | Visualization               |
| Pandas, NumPy    | Data handling               |

---

## 🚀 How to Run

1. Upload dataset to Google Drive or Colab environment.
2. Load and preprocess data (resize, normalize).
3. Train the model using the notebook:
   ```bash
   Lung_Cancer_Detection.ipynb
Evaluate performance on validation/test set.

Save trained model as .h5 or .pb.

📊 Evaluation Metrics
Accuracy

Precision, Recall

Confusion Matrix

Loss & Accuracy Curves


📈 Sample Results
Metric	Value
Accuracy	95.6%
Val Loss	Low
ROC AUC	~0.94
