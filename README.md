# Dog Emotion Classification Using CNNs, Transfer Learning & Ensemble Learning

A deep learning project for classifying dog images into four apparent emotional states — **Angry, Happy, Relaxed, and Sad** — using a custom CNN, multiple ImageNet-pretrained architectures, fine-tuning, and feature-level ensemble learning.

The project compares five pretrained CNN architectures and combines the two strongest models, **EfficientNetB0 and ResNet50**, into a feature-level ensemble that achieves **84.13% test accuracy**.

---

## Project Overview

Understanding animal behaviour from visual information is a challenging computer vision problem. Dogs communicate through facial expressions, posture, body orientation, and other visual cues, making automated emotion classification an interesting application of deep learning.

This project follows a progressive modelling approach:

1. Build a custom CNN baseline.
2. Establish a fixed train-validation-test split.
3. Apply image augmentation.
4. Benchmark multiple ImageNet-pretrained CNN architectures.
5. Fine-tune the pretrained models.
6. Compare models using validation and test performance.
7. Combine the strongest models using feature-level ensemble learning.
8. Evaluate the final model using accuracy, precision, recall, and F1-score.

---

## Problem Statement

Given an image containing a dog, predict its apparent emotional state as one of four classes:

* **Angry**
* **Happy**
* **Relaxed**
* **Sad**

This is formulated as a **four-class multiclass image classification problem**.

---

## Dataset

The dataset contains labelled dog images belonging to four emotion categories.

| Class   | Description                    |
| ------- | ------------------------------ |
| Angry   | Angry or aggressive expression |
| Happy   | Happy or positive expression   |
| Relaxed | Calm or relaxed expression     |
| Sad     | Sad or subdued expression      |

The final test set contains **586 images**:

| Emotion   | Test Images |
| --------- | ----------: |
| Angry     |         137 |
| Happy     |         156 |
| Relaxed   |         139 |
| Sad       |         154 |
| **Total** |     **586** |

The dataset is relatively balanced across the four classes.

---

## Train / Validation / Test Split

A fixed **70% / 15% / 15%** split was used throughout the project.

```text
Dataset
   │
   ├── 70% Training
   │      └── Model fitting
   │
   ├── 15% Validation
   │      └── Model selection / Early stopping
   │
   └── 15% Test
          └── Final evaluation
```

The test set was kept untouched during training and model selection to provide an unbiased final evaluation.

---

## Data Preprocessing & Augmentation

Images were resized according to the input requirements of each architecture.

### Input Sizes

| Model          | Input Size    |
| -------------- | ------------- |
| Custom CNN     | 160 × 160 × 3 |
| MobileNetV2    | 160 × 160 × 3 |
| EfficientNetB0 | 224 × 224 × 3 |
| ResNet50       | 224 × 224 × 3 |
| Xception       | 224 × 224 × 3 |
| InceptionV3    | 299 × 299 × 3 |

Training-time augmentation included:

* Random horizontal flipping
* Random rotation
* Random zoom

These transformations were used to improve model generalization and reduce overfitting.

---

# Model Development

## 1. Custom CNN Baseline

A custom CNN was developed from scratch to establish a baseline.

### Architecture

```text
Input Image (160 × 160 × 3)
          │
          ▼
   Data Augmentation
          │
          ▼
   Conv Block 1
     32 Filters
          │
          ▼
   Conv Block 2
     64 Filters
          │
          ▼
   Conv Block 3
    128 Filters
          │
          ▼
   Conv Block 4
    128 Filters
          │
          ▼
Global Average Pooling
          │
          ▼
 Dense Layer (128)
          │
          ▼
   Softmax (4 Classes)
```

Batch normalization, dropout, and L2 regularization were incorporated to improve optimization and reduce overfitting.

The baseline achieved:

**Test Accuracy: 61.95%**

This provided a reference point for measuring the benefit of transfer learning.

---

# 2. Transfer Learning

Five ImageNet-pretrained CNN architectures were benchmarked:

* **MobileNetV2**
* **EfficientNetB0**
* **ResNet50**
* **Xception**
* **InceptionV3**

The pretrained convolutional backbones were initially frozen while a new classification head was trained.

### Classification Head

```text
Pretrained Backbone
        │
        ▼
Global Average Pooling
        │
        ▼
     Dropout
        │
        ▼
Dense Layer (128)
        │
        ▼
     Dropout
        │
        ▼
Softmax Output (4 Classes)
```

---

# 3. Fine-Tuning

After training the classification head, the **final 30 layers** of each pretrained backbone were unfrozen.

Fine-tuning was performed using a smaller learning rate:

```text
Initial training LR = 1e-3
Fine-tuning LR      = 1e-5
```

This allowed the newly added classification layers to adapt first, followed by smaller updates to the pretrained visual representations.

---

# 4. Feature-Level Ensemble

The two strongest individual architectures were:

* **ResNet50**
* **EfficientNetB0**

Instead of simply averaging their predictions, their learned feature representations were combined.

```text
                 Dog Image
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    EfficientNetB0          ResNet50
      Backbone              Backbone
          │                   │
          ▼                   ▼
    Feature Vector        Feature Vector
          │                   │
          └─────────┬─────────┘
                    ▼
            Feature Concatenation
                    │
                    ▼
              Dense Layer
                    │
                    ▼
                 Dropout
                    │
                    ▼
             Softmax Output
                    │
                    ▼
       Angry | Happy | Relaxed | Sad
```

The combined feature representation is:

```text
f = [EfficientNetB0 features ; ResNet50 features]
```

The concatenated representation is then passed through additional dense layers before the final four-class softmax classifier.

---

# Results

## Overall Model Comparison

| Model          | Validation Accuracy | Test Accuracy |
| -------------- | ------------------: | ------------: |
| **Ensemble**   |          **85.07%** |    **84.13%** |
| ResNet50       |              83.68% |        82.94% |
| EfficientNetB0 |              83.85% |        82.76% |
| Xception       |              79.51% |        80.72% |
| InceptionV3    |              77.43% |        77.13% |
| MobileNetV2    |              75.35% |        73.21% |
| Baseline CNN   |              66.84% |        61.95% |

The **EfficientNetB0 + ResNet50 ensemble** achieved the highest test accuracy.

---

## Final Performance

The final ensemble achieved:

| Metric              |      Score |
| ------------------- | ---------: |
| Validation Accuracy | **85.07%** |
| Test Accuracy       | **84.13%** |
| Macro Recall        | **84.03%** |
| Macro F1-Score      | **84.10%** |
| Angry Recall        | **80.29%** |

The ensemble improved test accuracy by **1.19 percentage points** over the strongest individual model, ResNet50.

```text
ResNet50 Accuracy       = 82.94%
Ensemble Accuracy       = 84.13%

Improvement             = +1.19 percentage points
```

---

# Class-Level Performance

### EfficientNetB0 + ResNet50 Ensemble

| Class             | Precision |   Recall | F1-Score | Support |
| ----------------- | --------: | -------: | -------: | ------: |
| Angry             |      0.83 |     0.80 |     0.82 |     137 |
| Happy             |      0.85 |     0.87 |     0.86 |     156 |
| Relaxed           |      0.87 |     0.85 |     0.86 |     139 |
| Sad               |      0.81 |     0.84 |     0.82 |     154 |
| **Macro Average** |  **0.84** | **0.84** | **0.84** | **586** |

The ensemble performed particularly well on the **Happy** and **Relaxed** classes, both achieving an F1-score of **0.86**.

---

# Key Findings

* The custom CNN achieved **61.95%** test accuracy.
* Transfer learning substantially improved performance over training a CNN from scratch.
* **ResNet50** was the strongest individual model with **82.94%** test accuracy.
* **EfficientNetB0** closely followed with **82.76%** test accuracy.
* Xception achieved **80.72%** test accuracy.
* InceptionV3 achieved **77.13%** test accuracy.
* MobileNetV2 achieved **73.21%** test accuracy.
* The **EfficientNetB0 + ResNet50 feature-level ensemble** achieved the best performance.
* The final ensemble achieved **84.13% test accuracy** and **84.10% macro F1-score**.
* The ensemble improved upon ResNet50 by **1.19 percentage points**.
* The final model achieved **80.29% recall for the Angry class**.

---

# Tech Stack

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **CNNs**
* **Transfer Learning**
* **Fine-Tuning**
* **Ensemble Learning**

---

# Project Structure

A recommended repository structure is:

```text
dog-emotion-classification/
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── notebooks/
│   └── dog_emotion_classification.ipynb
│
├── models/
│   ├── baseline_cnn/
│   ├── mobilenetv2/
│   ├── efficientnetb0/
│   ├── resnet50/
│   ├── xception/
│   ├── inceptionv3/
│   └── ensemble/
│
├── results/
│   ├── training_curves/
│   ├── confusion_matrix/
│   └── model_comparison/
│
├── requirements.txt
└── README.md
```

---

# Training Strategy

The complete modelling pipeline can be summarized as:

```text
                Dog Emotion Dataset
                         │
                         ▼
                70 / 15 / 15 Split
                         │
                         ▼
              Preprocessing & Augmentation
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Custom CNN             Pretrained CNNs
             │                       │
             │        ┌──────────────┼──────────────┐
             │        ▼              ▼              ▼
             │   MobileNetV2   EfficientNetB0   ResNet50
             │        │              │              │
             │        └──────┬───────┴──────┬───────┘
             │               │              │
             │               ▼              ▼
             │          Xception       InceptionV3
             │
             ▼
        Baseline Result
                         │
                         ▼
                  Fine-Tuning
                         │
                         ▼
             Select Strongest Models
                         │
                         ▼
           EfficientNetB0 + ResNet50
                         │
                         ▼
              Feature Concatenation
                         │
                         ▼
                 Final Classifier
                         │
                         ▼
             4-Class Emotion Prediction
```

---

# Limitations

Despite the strong experimental performance, the project has several limitations:

1. **Subjectivity of emotion labels**
   Emotional state cannot always be reliably determined from a static image.

2. **Image-only information**
   The models cannot use temporal behavioural information available in videos.

3. **Dataset limitations**
   Performance depends on the diversity, quality, and representativeness of the dataset.

4. **Domain generalization**
   Performance may decrease on images with substantially different backgrounds, lighting, breeds, or poses.

5. **Interpretability**
   CNN predictions do not inherently explain which image regions influenced the prediction.

---

# Future Work

Potential extensions include:

* Applying **Grad-CAM** for visual interpretability.
* Expanding the dataset across more breeds, poses, lighting conditions, and environments.
* Exploring **video-based emotion recognition**.
* Investigating attention-based CNN architectures and Vision Transformers.
* Performing external validation on an independently collected dataset.
* Developing class-specific augmentation strategies.
* Optimizing the ensemble for deployment on resource-constrained devices.

---

# Conclusion

This project demonstrates a complete deep learning pipeline for four-class dog emotion classification.

A custom CNN was first developed as a baseline and achieved **61.95% test accuracy**. Five ImageNet-pretrained architectures were subsequently evaluated using transfer learning and fine-tuning, with all pretrained models substantially outperforming the custom CNN.

Among the individual models, **ResNet50 achieved 82.94% test accuracy**, followed closely by **EfficientNetB0 at 82.76%**.

These two models were combined using feature-level ensemble learning. The resulting **EfficientNetB0 + ResNet50 ensemble achieved 84.13% test accuracy, 84.03% macro recall, and 84.10% macro F1-score**.

Overall, the results demonstrate that **transfer learning, fine-tuning, and feature-level ensemble learning can provide a substantially stronger solution than training a CNN entirely from scratch for dog emotion classification.**

---

## Final Summary

| Component                    | Result                             |
| ---------------------------- | ---------------------------------- |
| **Task**                     | 4-Class Dog Emotion Classification |
| **Classes**                  | Angry, Happy, Relaxed, Sad         |
| **Dataset Split**            | 70% / 15% / 15%                    |
| **Baseline**                 | Custom CNN                         |
| **Baseline Accuracy**        | 61.95%                             |
| **Best Individual Model**    | ResNet50                           |
| **Best Individual Accuracy** | 82.94%                             |
| **Final Model**              | EfficientNetB0 + ResNet50 Ensemble |
| **Final Test Accuracy**      | **84.13%**                         |
| **Macro Recall**             | **84.03%**                         |
| **Macro F1**                 | **84.10%**                         |
| **Angry Recall**             | **80.29%**                         |
| **Ensemble Improvement**     | **+1.19 percentage points**        |

---

## Author

**Vaibhav Khare**

Deep Learning & Computer Vision Project — 2026
