# 🧠 Deep Learning Models for Brain Tumor Classification with Explainable AI

*Author:* Bhuvankumar A Patri  
*Architecture:* ConvNeXt, Swin Transformer, EfficientNet-B0  
*Optimization:* Aquila Optimizer (AQ), Random Search (RS), Baseline  
*Explainability:* Grad-CAM++, LIME, SHAP  

---

## 📌 ABSTRACT

Accurate and early diagnosis of brain tumors is crucial for effective clinical intervention and improved patient survival rates. Although Magnetic Resonance Imaging (MRI) is widely used for detection, manual interpretation is time-consuming and subject to inter-observer variability. This work presents a deep learning framework for multi-class brain tumor classification using a unified preprocessing pipeline and systematic experimental evaluation. All experiments are conducted on the Kaggle Brain Tumor MRI dataset, consisting of 7,023 T1-weighted contrast-enhanced MRI images across four classes: glioma, meningioma, pituitary tumor, and no tumor. A total of seven controlled experiments are performed across three modern architectures—ConvNeXt, Swin Transformer, and EfficientNet-B0—combined with Aquila Optimizer (AQ), Random Search (RS), and baseline configurations. All models achieve high classification performance exceeding 99% accuracy, with observable differences in confidence calibration measured using mean Average Precision (mAP). Among all configurations, ConvNeXt optimized with AQ achieves the best performance with 99.69% accuracy, MCC of 0.9959, and a perfect mAP of 1.0000, indicating superior ranking confidence and decision boundary refinement. Other configurations, including Swin Transformer (RS and AQ) and EfficientNet-B0 (AQ), demonstrate competitive performance, validating architectural robustness. A multi-level Explainable AI (XAI) framework integrating Grad-CAM++, LIME, and SHAP is applied exclusively to the best-performing ConvNeXt + AQ model to ensure interpretability. A Flask-based diagnostic dashboard is developed to demonstrate real-time prediction and explainability capabilities.
---

## 🚀 KEY CONTRIBUTIONS

- Controlled comparison of ConvNeXt, Swin Transformer, EfficientNet-B0  
- Metaheuristic optimization using Aquila Optimizer (AQ)  
- Validation against Random Search and baseline models  
- Achievement of *perfect mAP (1.0000)*  
- Multi-level XAI (Grad-CAM++, LIME, SHAP)  
- End-to-end Flask-based diagnostic system  

---

## 🔬 METHODOLOGY

The proposed framework follows a consistent pipeline across all seven experiments, ensuring fair comparison between architectures and optimization strategies.

### 1. Dataset Preparation
- Dataset loaded using `ImageFolder` from the Kaggle Brain Tumor MRI dataset
- Training and testing directories used separately (`Training/`, `Testing/`)
- Class labels automatically extracted from folder structure

---

### 2. Data Preprocessing & Augmentation
A unified preprocessing pipeline is applied across all models:

**Training Transformations:**
- Resize to 224 × 224  
- Random Horizontal Flip  
- Random Rotation (±20°)  
- Color Jitter (brightness & contrast)  
- Normalization using ImageNet mean & std  

**Validation/Test Transformations:**
- Resize to 224 × 224  
- Normalization only (no augmentation)

---

### 3. Data Splitting Strategy

Two evaluation strategies were used:

- **Single Split (Stratified):**
  - 80% training, 20% validation  
  - Stratified using `train_test_split` to maintain class balance  

- **5-Fold Cross-Validation:**
  - Dataset split into 5 folds using `KFold`
  - Each fold used once as validation set  
  - Final performance averaged across folds  

---

### 4. Model Architectures

Three architectures were evaluated:

- **ConvNeXt-Tiny** (pretrained on ImageNet)  
- **Swin Transformer (Tiny)**  
- **EfficientNet-B0**  

For all models:
- Final classification layer modified to 4 classes  
- Dropout layer added/tuned for regularization  

---

### 5. Training Configuration

- Loss Function: CrossEntropyLoss  
- Optimizer: AdamW  
- Mixed Precision Training using `torch.cuda.amp`  
- Batch size: ~26–32 (varies by experiment)  
- Epochs: 20  

---

### 6. Hyperparameter Optimization

Three strategies were used:

- **Baseline:** Fixed hyperparameters  
- **Random Search (RS):** Random sampling of parameters  
- **Aquila Optimizer (AQ):** Metaheuristic optimization  

AQ optimizes:
- Learning rate  
- Dropout  
- Batch size  

Example (ConvNeXt + AQ):
- Learning Rate ≈ 1.33e-4  
- Dropout ≈ 0.27  
- Batch Size ≈ 26  

---

### 7. Evaluation Metrics

Performance evaluated using:

- **Accuracy** → Overall correctness  
- **MCC (Matthews Correlation Coefficient)** → Balanced metric  
- **mAP (mean Average Precision)** → Confidence calibration  
- **Confusion Matrix & Classification Report**  
- **Class-wise Recall** (critical for medical evaluation)  

---

### 8. Inference & Deployment

- Best model (ConvNeXt + AQ) selected  
- Integrated into Flask-based web application  
- Real-time prediction with XAI outputs  

---

### 🔁 Overall Pipeline

```
Data → Preprocessing → Split → Model Training → Optimization → Evaluation → Deployment
```

## 📊 EXPERIMENTAL RESULTS (N = 1,311)

| Exp | Model Configuration| Optimizer| Accuracy (%) | MCC     | mAP      | Errors |
|-----|--------------------|----------|--------------|---------|----------|--------|
| 01  | ConvNeXt (Single)  | Aquila   | 99.69        | 0.9959  |**1.0000**|    4   |
| 02  | ConvNeXt (Single)  | Baseline | 99.69        | 0.9959  |  0.9999  |    4   |
| 03  | Swin Transformer   | RS       | 99.62        | 0.9949  |  0.9999  |    5   |
| 04  | Swin Transformer   | Aquila   | 99.62        | 0.9949  |  0.9995  |    5   |
| 05  | ConvNeXt (K-Fold)  | Aquila   | 99.47        | 0.9928  |  0.9998  |    7   |
| 06  | EfficientNet-B0    | Aquila   | 99.47        | 0.9929  |  0.9999  |    7   |
| 07  | ConvNeXt (K-Fold)  | Baseline | 99.24        | 0.9898  |  0.9997  |    10  |

---

### 🔍 Key Observations

- **ConvNeXt + AQ (Single & K-Fold)** and **Swin Transformer (RS & AQ)** achieved **100% recall for the Meningioma tumor class**, indicating that these models consistently detect all meningioma cases without false negatives.

- **EfficientNet-B0 + AQ** achieved **100% recall for the No Tumor class**, demonstrating strong reliability in correctly identifying healthy cases.

- These results highlight that while overall accuracy is similar across models, **class-wise performance varies**, emphasizing the importance of per-class evaluation in medical diagnosis tasks.

- Achieving 100% recall in specific classes is particularly significant in clinical settings, as it ensures **no critical cases are missed**.

- **Model Behavior Insight:** ConvNeXt + AQ demonstrates stronger performance in **tumor detection**, while EfficientNet-B0 shows higher reliability in **normal (no tumor) classification**, indicating a degree of **model specialization across classes**.

---

💡 **Why Recall Matters (Clinical Perspective)**  
In medical diagnosis, recall is critical because missing a tumor (false negative) is more dangerous than a false positive. Achieving 100% recall ensures that no actual tumor cases are overlooked, making the model safer for clinical decision support.

## 🧠 EXPLAINABLE AI (XAI)

Applied *only to ConvNeXt + AQ (best model)*:

- Grad-CAM++ → Tumor localization  
- LIME → Local feature importance  
- SHAP → Pixel-level contribution  

Ensures *clinical interpretability + trustworthiness*

---

## 🌐 WEB APPLICATION

Flask-based diagnostic dashboard:

- Secure login system  
- MRI upload interface  
- Real-time prediction  
- Explainability visualization  
- Clinical-style feedback  
- PDF report generation  

---

## 📂 COMPLETE PROJECT STRUCTURE

```
brain-tumor-classification-xai/
│
├── model/                          # Final trained model
│   └── final_convnext_aq.pth
│
├── model_experiments/              # Research experiments (CORE)
│   ├── baseline-convnext.ipynb
│   ├── convnext-aq.ipynb
│   ├── convnext-baseline-k-fold.ipynb
│   ├── efficientnet-b0-aq.ipynb
│   ├── k-fold-convnext-aq.ipynb
│   ├── SWIN-T+AO.ipynb
│   └── swin-t-rs.ipynb
│
├── static/                         # Frontend + runtime assets
│   ├── css/
│   ├── js/
│   ├── background.jpg
│   ├── confusion_matrix_train.png
│   ├── confusion_matrix_test.png
│   ├── favicon.ico
│   ├── nmit_logo.jpeg
│   └── outputs/                   # Uploaded MRI + generated XAI outputs
│
├── templates/                      # HTML templates
│   ├── base.html
│   ├── index1.html
│   ├── login.html
│   └── register.html
│
├── utils/                          # Core logic
│   └── explainer1.py
│
├── app1.py                         # Flask backend
├── evaluate_dataset.py            # Evaluation script
├── generate_train_features.py     # Feature extraction
│
├── results_train.csv              # Training metrics lookup
├── results_test.csv               # Testing metrics lookup
│
├── requirements.txt
├── runtime.txt
└── users.db
```


---

## 🧠 STRUCTURE EXPLANATION

- *model_experiments/* → Research backbone (7 experiments)
- *model/* → Final selected model
- *utils/* → Explainability logic
- *app1.py* → Full web app backend
- *templates/static/* → UI
- *uploads/* → User inputs

---

## 🔁 SYSTEM PIPELINE


Experiments → Model Selection → Flask App → Upload → Prediction → XAI → Report


---

## ⚙️ INSTALLATION & LOCAL EXECUTION

```bash
git clone https://github.com/gangabhuvan/brain-tumor-classification-xai.git
cd brain-tumor-classification-xai

pip install -r requirements.txt

python app1.py
```

Open in browser:
```
http://127.0.0.1:5000
```

---

## 📌 NOTES

- Research-heavy → notebooks dominate language stats  
- Production uses best model only  
- Model auto-downloads if not present  

---

## 📚 FUTURE WORK

- Faster inference (async processing)  
- Cloud deployment optimization  
- Clinical validation  

---

## 🏁 CONCLUSION

ConvNeXt + Aquila Optimizer achieves superior performance with perfect mAP, while multi-level explainability ensures transparency, making the system suitable for real-world clinical decision support.

---
