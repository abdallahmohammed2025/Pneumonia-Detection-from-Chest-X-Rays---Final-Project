# Pneumonia Detection from Chest X-Rays using CNN + Grad-CAM

This project aims to automatically detect pneumonia from chest X-ray images using a Convolutional Neural Network (CNN) and provide interpretability for the predictions via Grad-CAM visualizations.

---

## Project Overview

Pneumonia is a serious lung infection that requires prompt diagnosis. Chest X-rays are a standard diagnostic tool, but interpretation requires expert radiologists and can be time-consuming. This project leverages deep learning for automated pneumonia detection, offering a scalable and reliable solution.

The CNN model is trained on labeled chest X-ray images to classify between normal and pneumonia cases. Grad-CAM is used to visualize which areas of the X-rays influence the model’s predictions, improving clinical trust and model transparency.

---

## Dataset

- The dataset used is the Chest X-Ray Images (Pneumonia) dataset.
- The dataset is organized into three subsets: `train`, `val`, and `test`.
- Each subset contains two categories: `NORMAL` and `PNEUMONIA`.
- Example counts for each subset are printed to verify dataset distribution.

---

## Methodology

### Data Loading and Exploration

- Dataset directories are traversed to count images per class.
- Sample images from the pneumonia class are displayed for visual inspection.

### Data Preprocessing

- Images are resized to 150x150 pixels.
- Data augmentation applied during training: rescaling, zoom, and horizontal flipping.
- Validation and test images are only rescaled.

### Model Architecture

- A sequential CNN model with three convolutional layers:
  - Conv2D layers with 32, 64, and 128 filters respectively, each followed by max-pooling.
- Followed by a flatten layer, dropout (0.5), a dense layer with 128 units, and a sigmoid output for binary classification.
- Compiled with Adam optimizer and binary cross-entropy loss.

### Training

- The model is trained for 10 epochs.
- `ModelCheckpoint` callback saves the best model by validation accuracy.
- Early stopping and training progress monitoring are included.

### Evaluation

- The best model is loaded and evaluated on the test set.
- Test accuracy and other metrics like confusion matrix and classification report are generated.

---

## Grad-CAM Visualization

- Grad-CAM heatmaps are generated to highlight areas of the X-ray that most influenced the model’s prediction.
- The last convolutional layer is identified dynamically for Grad-CAM calculation.
- Heatmaps are superimposed on the original X-ray images for visual interpretation.
- This helps clinicians verify that the model focuses on relevant regions for pneumonia detection.

---

## Results

- The CNN achieved approximately 95% validation accuracy.
- Test accuracy reached approximately 94.7%.
- Grad-CAM visualizations successfully highlight pneumonia-affected lung regions.

---

## Conclusion and Future Work

- The CNN model effectively automates pneumonia detection from chest X-rays.
- Grad-CAM provides valuable interpretability critical for medical adoption.
- Future improvements could include:
  - Applying transfer learning with pretrained architectures like DenseNet or Inception.
  - Enhancing Grad-CAM visualization techniques.
  - Deploying the model as a web-based clinical decision support tool.

---


## GitHub Repository:
[https://github.com/abdallahmohammed2025/Pneumonia-Detection-from-Chest-X-Rays---Final-Project](https://github.com/abdallahmohammed2025/Pneumonia-Detection-from-Chest-X-Rays---Final-Project)
 

---

## Usage

1. Prepare the dataset in the specified directory structure (`train`, `val`, `test` folders with subfolders `NORMAL` and `PNEUMONIA`).
2. Run the notebook to train and evaluate the CNN model.
3. Generate Grad-CAM heatmaps for test images to interpret model decisions.
4. Use the trained model for pneumonia classification in new X-ray images.

---

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Seaborn
- scikit-learn

---

## References

- [Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.

