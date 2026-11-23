# Skin Lesion Classification Using Classical Machine Learning and Deep Learning on the HAM10000 Dataset

An academic machine learning and deep learning project (**XBDS3024N**) comparing classical feature-based classification and deep transfer learning under baseline and class-imbalance-handled conditions on dermatoscopic skin lesion images from the **HAM10000** dataset[cite: 3, 4]. It also includes an interactive **Streamlit** web application for inference[cite: 1].

> **Disclaimer:** This project is built for academic and educational purposes only and is **not** a medical device or a substitute for professional clinical diagnosis.

---

## Project Overview & Class Imbalance Handling

Dermatoscopic datasets like HAM10000 exhibit strong class imbalance (dominated by *Melanocytic nevi* with fewer samples in rare categories such as *Dermatofibroma* and *Vascular lesions*)[cite: 3]. Rather than modifying the dataset via resampling (such as SMOTE), this project investigates **cost-sensitive learning**:

* **Baseline Random Forest (No Imbalance Handling):** Trained without class weighting, resulting in high overall test accuracy (`69%`) driven by majority-class dominance, poor *Melanoma* recall (`9%`, 3/34 cases), and zero-metric outcomes on rare classes (`0.00` F1-score on *Dermatofibroma* and *Vascular lesions*)[cite: 1, 2].
* **Balanced Random Forest (`class_weight='balanced'`):** Applies cost-sensitive class weighting during tree training to penalize minority-class misclassifications more heavily, improving *Melanoma* recall (`41%`, 14/34 cases) and lifting Macro F1-score to `0.4513` (`62%` test accuracy)[cite: 2, 4].
* **MobileNetV2 Transfer Learning (Class-Weighted Cross-Entropy + Augmentations):** Fine-tuned over 10 epochs (`128×128` inputs, ImageNet normalization, and training augmentations) with class-weighted cross-entropy loss, achieving the strongest Macro F1-score (`0.4920`, rounded to `0.49` in the report table) and `67%` test accuracy across 301 test samples[cite: 3, 4].

---

## Model Comparison Summary

| Model / Approach | Imbalance Handling | Test Accuracy | Macro F1-Score |
| :--- | :--- | :---: | :---: |
| **Baseline Random Forest** | None (unweighted) | `69%` | `0.2607`[cite: 4] |
| **Balanced Random Forest** | Cost-sensitive (`class_weight='balanced'`)[cite: 2] | `62%`[cite: 2] | `0.4513`[cite: 4] |
| **MobileNetV2 (PyTorch)** | Weighted cross-entropy loss + augmentations | `67%` (`0.67`)[cite: 3] | `0.4920` (`0.49` rounded)[cite: 3, 4] |

### Latest MobileNetV2 Highlights (301 Test Samples)
* **Overall Accuracy:** `67%` (`0.67`)[cite: 3]
* **Macro F1-Score:** `0.4920` (`0.49` in classification report table)[cite: 3, 4]
* **Weighted Avg F1-Score:** `0.70`[cite: 3]
* **Melanocytic nevi:** `0.96` precision, `0.74` recall, `0.84` F1-score (`201` support)[cite: 3]
* **Vascular lesions:** `0.40` precision, `1.00` recall, `0.57` F1-score (`4` support)[cite: 3]
* **Melanoma:** `0.27` precision, `0.41` recall, `0.33` F1-score (`34` support)[cite: 3]
* **Dermatofibroma:** `0.14` precision, `0.33` recall, `0.20` F1-score (`3` support)[cite: 3]

---

## Repository Structure

```text
├── app.py                  # Streamlit web application for skin lesion prediction
├── *.ipynb                 # Jupyter notebook(s) for preprocessing, training, and evaluation
├── mobilenetv2_skin.pth    # Saved PyTorch MobileNetV2 model weights
├── mobilenetv2_report.png  # MobileNetV2 classification report visual[cite: 3]
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and dataset directories
└── README.md               # Project documentation

Setup & Usage
1. Clone the repository:

Bash
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>

2. Install dependencies:

Bash
pip install -r requirements.txt

3. Run the Streamlit app:

Bash
streamlit run app.py