# Monkeypox Detection Using Deep Learning

## 📌 Project Overview
This project presents an automated medical image classification system for detecting **Monkeypox skin lesions** using deep learning techniques. The system compares the performance of **CNN, ResNet50, and MobileNet** architectures and deploys the best-performing model using a **Flask web application**.

## 🧠 Models Used
- Custom Convolutional Neural Network (CNN)
- ResNet50 (Transfer Learning + Fine-Tuning)
- MobileNet (Transfer Learning + Fine-Tuning)

## 🗂 Dataset
- Medical images of Monkeypox and normal skin
- Images were preprocessed and augmented
- Dataset split into training and validation sets
- Due to size and privacy constraints, the full dataset is not included in this repository

## ⚙️ Methodology
- Image preprocessing and normalization
- Data augmentation using ImageDataGenerator
- Transfer learning with fine-tuning
- Model evaluation using accuracy and loss metrics
- Deployment using Flask

## 📊 Results
| Model       | Performance |
|------------|-------------|
| CNN        | Improved after fine-tuning |
| ResNet50  | Good generalization |
| MobileNet | Best accuracy and efficiency |

MobileNet achieved the best balance between accuracy and computational efficiency.

## 🖥 Web Application
- Built using Flask
- Allows users to upload skin lesion images
- Predicts whether the image indicates Monkeypox
## 📸 Application Screenshots

### Home Page
![Home Page](screenshots/homepage.png)

### Upload Page
![Upload Page](screenshots/img_select.png)

### Prediction Result
![Result](screenshots/MPox_detect.png)
![Result](screenshots/MPox_detect2.png)
![Result](screenshots/No_MPox.png)


## 🛠 Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- Flask
- NumPy, Matplotlib

## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
python app.py
