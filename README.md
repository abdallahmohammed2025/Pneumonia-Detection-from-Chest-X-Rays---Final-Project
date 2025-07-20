# Pneumonia Detection from Chest X-Rays (Final Project)

This repository contains the code and resources for the final deep learning project focused on detecting pneumonia from chest X-ray images using convolutional neural networks (CNNs) and Grad-CAM for model explainability.

---

## Project Overview

Pneumonia is a serious respiratory infection that can be life-threatening if not diagnosed early and treated properly. This project aims to build an automated system that detects pneumonia from chest X-rays leveraging deep learning techniques.

The core components of this project include:

- **Exploratory Data Analysis (EDA):** Understand and visualize the dataset, including the distribution of normal and pneumonia cases.
- **Data Preprocessing:** Image resizing, normalization, and augmentation.
- **Model Building:** A CNN architecture trained to classify chest X-ray images as pneumonia or normal.
- **Training and Evaluation:** Model training, validation, and evaluation on test data.
- **Explainability:** Using Grad-CAM to visualize the regions of the X-rays that influence the model’s decisions.

---

## Dataset

The dataset used is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset, which contains labeled chest X-ray images of pneumonia and normal cases.

---

## Getting Started

### Prerequisites

- Python 3.7 or later
- TensorFlow 2.x
- kagglehub (for automated dataset download)
- Other standard Python libraries such as numpy, matplotlib, opencv-python

### Installation

You can install the required Python packages via pip:

pip install tensorflow kagglehub matplotlib numpy opencv-python

---

## Usage

1. Clone this repository:

git clone https://github.com/abdallahmohammed2025/Pneumonia-Detection-from-Chest-X-Rays---Final-Project.git
cd Pneumonia-Detection-from-Chest-X-Rays---Final-Project


2. Open the Jupyter notebook `Pneumonia_Detection.ipynb` in your preferred environment (VSCode, Jupyter Notebook, etc.).

3. Run the notebook cells sequentially. The notebook includes code to download and extract the dataset automatically using `kagglehub`.

4. Train the CNN model and observe the training process, evaluation metrics, and Grad-CAM visualizations.

---

## Results

- The CNN model achieves reasonable accuracy in detecting pneumonia from chest X-rays.
- Grad-CAM visualizations highlight critical regions contributing to the model’s predictions, helping interpretability.
- The project notebook contains plots and discussions on training/validation loss, accuracy, and example Grad-CAM outputs.

---

## Repository Structure

- `Pneumonia_Detection.ipynb` — Jupyter notebook containing the full pipeline from data loading to model training and visualization.
- `README.md` — Project overview and instructions.

---

## Acknowledgments

- Dataset by [Paul Timothy Mooney](https://www.kaggle.com/paultimothymooney)
- Inspired by deep learning best practices for medical image analysis.

---

## License

This project is for educational purposes.

---

## Contact

For questions or feedback, please open an issue or contact the repository owner.

---

**GitHub Repository:**  
[https://github.com/abdallahmohammed2025/Pneumonia-Detection-from-Chest-X-Rays---Final-Project](https://github.com/abdallahmohammed2025/Pneumonia-Detection-from-Chest-X-Rays---Final-Project)