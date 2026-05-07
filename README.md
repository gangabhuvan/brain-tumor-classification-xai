````markdown
# Aquila-Optimized Deep Learning for Brain Tumor Classification: A Comparative Study of Optimization Strategies, Model Stability, and Explainable AI

🎓 A focused collaborative version of this research based on the EfficientNet-B0 + Aquila Optimizer framework has been accepted for Oral Presentation and Publication at ICT4SD 2026 (Springer LNNS Series).

*A research-driven deep learning framework for brain tumor classification with metaheuristic optimization and clinically interpretable AI.*

**Primary Research & Implementation:** BHUVANKUMAR A. PATRI

- **Architecture:** ConvNeXt, Swin Transformer, EfficientNet-B0  
- **Optimization:** Aquila Optimizer (AO), Random Search (RS)  
- **Baseline:** Default (non-optimized) configuration  
- **Explainability:** Grad-CAM++, LIME, SHAP  

---

# 📌 ABSTRACT

Accurate and early diagnosis of brain tumors is crucial for effective clinical intervention and improved patient survival rates. Although Magnetic Resonance Imaging (MRI) is widely used for detection, manual interpretation is time-consuming and subject to inter-observer variability.

This work presents a deep learning framework for multi-class brain tumor classification using a unified preprocessing pipeline and systematic experimental evaluation.

All experiments are conducted on the Kaggle Brain Tumor MRI dataset, consisting of 7,023 T1-weighted contrast-enhanced MRI images across four classes: glioma, meningioma, pituitary tumor, and no tumor.

A total of seven controlled experiments are performed across three modern architectures—ConvNeXt, Swin Transformer, and EfficientNet-B0—combined with Aquila Optimizer (AO), Random Search (RS), and baseline configurations.

All models achieve consistently high performance (>99% accuracy), indicating strong feature separability in MRI-based tumor classification with observable differences in confidence calibration measured using mean Average Precision (mAP).

Among all configurations, ConvNeXt optimized with AO achieves the best performance with 99.69% accuracy, MCC of 0.9959, and a maximum observed mAP of 1.0000, indicating superior ranking confidence and decision boundary refinement.

Other configurations, including Swin Transformer (RS and AO) and EfficientNet-B0 (AO), demonstrate competitive performance, validating architectural robustness.

A multi-level Explainable AI (XAI) framework integrating Grad-CAM++, LIME, and SHAP is applied exclusively to the best-performing ConvNeXt + AO model to ensure interpretability.

A Flask-based diagnostic dashboard is developed to demonstrate real-time prediction and explainability capabilities.

---

# 🚀 KEY CONTRIBUTIONS

- Controlled comparison of ConvNeXt, Swin Transformer, and EfficientNet-B0  
- Metaheuristic optimization using Aquila Optimizer (AO)  
- Validation against Random Search and baseline models  
- Achievement of a maximum observed mAP of 1.0000  
- Multi-level Explainable AI (Grad-CAM++, LIME, SHAP)  
- End-to-end Flask-based diagnostic system  
- Comparative evaluation of ConvNeXt, Swin Transformer, and EfficientNet-B0 combined with AO and Random Search for brain tumor classification  

---

# 🔬 METHODOLOGY

The proposed framework follows a consistent pipeline across all seven experiments, ensuring fair comparison between architectures and optimization strategies.

## 1. Dataset Preparation

- Dataset loaded using `ImageFolder` from the Kaggle Brain Tumor MRI dataset  
- Training and testing directories used separately (`Training/`, `Testing/`)  
- Class labels automatically extracted from folder structure  

---

## 2. Data Preprocessing & Augmentation

A unified preprocessing pipeline is applied across all models.

### Training Transformations

- Resize to 224 × 224  
- Random Horizontal Flip  
- Random Rotation (±20°)  
- Color Jitter (brightness & contrast)  
- Normalization using ImageNet mean & standard deviation  

### Validation/Test Transformations

- Resize to 224 × 224  
- Normalization only (no augmentation)  

---

## 3. Data Splitting Strategy

Two evaluation strategies were used.

### Single Split (Stratified)

- 80% training, 20% validation  
- Stratified using `train_test_split` to maintain class balance  

### 5-Fold Cross-Validation

- Dataset split into 5 folds using `KFold`  
- Each fold used once as validation set  
- Final performance averaged across folds  

---

## 4. Model Architectures

Three architectures were evaluated.

- **ConvNeXt-Tiny** (pretrained on ImageNet)  
- **Swin Transformer (Tiny)**  
- **EfficientNet-B0**  

For all models:

- Final classification layer modified to 4 classes  
- Dropout layer added/tuned for regularization  

---

## 5. Training Configuration

- Loss Function: `CrossEntropyLoss`  
- Optimizer: `AdamW`  
- Mixed Precision Training using `torch.cuda.amp` for improved computational efficiency and reduced GPU memory usage  
- Batch size: ~26–32 (varies by experiment)  
- Training conducted for up to 20 epochs using `CosineAnnealingLR` scheduling with early stopping to prevent overfitting  

---

## 6. Hyperparameter Optimization

Three optimization strategies were used.

- **Baseline:** Fixed hyperparameters  
- **Random Search (RS):** Random parameter sampling  
- **Aquila Optimizer (AO):** Metaheuristic optimization  

AO optimizes:

- Learning rate  
- Dropout probability  
- Batch size  

Example (ConvNeXt + AO):

- Learning Rate ≈ 1.33e-4  
- Dropout ≈ 0.27  
- Batch Size ≈ 26  

---

## 7. Evaluation Metrics

Performance evaluated using:

- **Accuracy** → Overall correctness  
- **MCC (Matthews Correlation Coefficient)** → Balanced predictive reliability  
- **mAP (mean Average Precision)** → Confidence calibration and ranking quality  
- **Confusion Matrix & Classification Report**  
- **Class-wise Recall** (critical for medical evaluation)  

---

## 8. Inference & Deployment

- Best model (ConvNeXt + AO) selected  
- Integrated into a Flask-based web application  
- Real-time prediction with XAI outputs  

---

# 🔁 OVERALL PIPELINE

```text
Dataset (ImageFolder)
        ↓
Preprocessing & Augmentation
        ↓
Data Splitting (Stratified / K-Fold)
        ↓
Model Initialization
(ConvNeXt / Swin / EfficientNet)
        ↓
Hyperparameter Optimization
(AO / RS)
        ↓
Final Model Training
        ↓
Evaluation
(Accuracy, MCC, mAP, Recall, Confusion Matrix)
        ↓
Best Model Selection
(ConvNeXt + AO)
        ↓
Deployment
(Flask Application + XAI)
````

---

# 📊 EXPERIMENTAL RESULTS (N = 1,311)

| Exp | Model Configuration | Optimizer | Accuracy (%) | MCC    | mAP        | Errors |
| --- | ------------------- | --------- | ------------ | ------ | ---------- | ------ |
| 01  | ConvNeXt (Single)   | AO        | 99.69        | 0.9959 | **1.0000** | 4      |
| 02  | ConvNeXt (Single)   | Baseline  | 99.69        | 0.9959 | 0.9999     | 4      |
| 03  | Swin Transformer    | RS        | 99.62        | 0.9949 | 0.9999     | 5      |
| 04  | Swin Transformer    | AO        | 99.62        | 0.9949 | 0.9995     | 5      |
| 05  | ConvNeXt (K-Fold)   | AO        | 99.47        | 0.9928 | 0.9998     | 7      |
| 06  | EfficientNet-B0     | AO        | 99.47        | 0.9929 | 0.9999     | 7      |
| 07  | ConvNeXt (K-Fold)   | Baseline  | 99.24        | 0.9898 | 0.9997     | 10     |

---

# 🔍 KEY OBSERVATIONS

* **ConvNeXt + AO** and **Swin Transformer (RS & AO)** achieved **100% recall for the Meningioma class**, indicating no missed meningioma cases.

* **EfficientNet-B0 + AO** achieved **100% recall for the No Tumor class**, demonstrating strong reliability in healthy-case detection.

* Although overall accuracy remains similarly high across models, class-wise behavior differs, emphasizing the importance of per-class medical evaluation.

* ConvNeXt + AO demonstrates stronger tumor-detection behavior, while EfficientNet-B0 shows stronger reliability for normal-case classification.

* ROC-AUC values approach saturation (~1.0) across all models, making mAP and MCC more informative metrics for confidence calibration and prediction reliability.

---

# 💡 WHY RECALL MATTERS

In medical diagnosis, recall is critical because false negatives may result in missed tumor cases. High recall ensures safer clinical decision-support behavior and reduces the probability of overlooking clinically significant abnormalities.

---

# 🧠 EXPLAINABLE AI (XAI)

Applied exclusively to the best-performing ConvNeXt + AO model.

* **Grad-CAM++** → Tumor localization
* **LIME** → Local feature importance
* **SHAP** → Pixel-level contribution analysis

These methods improve transparency, interpretability, and clinical trustworthiness.

---

# 🌐 WEB APPLICATION

Flask-based diagnostic dashboard with:

* Secure login system
* MRI upload interface
* Real-time prediction
* Explainability visualization
* Clinical-style feedback
* PDF report generation

---

# 📂 REPOSITORY STRUCTURE

```text
brain-tumor-classification-xai/
│
├── model/
│   └── final_convnext_ao.pth
│
├── model_experiments/
│   ├── baseline-convnext.ipynb
│   ├── convnext-ao.ipynb
│   ├── convnext-baseline-k-fold.ipynb
│   ├── efficientnet-b0-ao.ipynb
│   ├── k-fold-convnext-ao.ipynb
│   ├── SWIN-T+AO.ipynb
│   └── swin-t-rs.ipynb
│
├── static/
├── templates/
├── utils/
├── app1.py
├── evaluate_dataset.py
├── generate_train_features.py
├── results_train.csv
├── results_test.csv
├── requirements.txt
├── runtime.txt
└── users.db
```

---

# ⚙️ INSTALLATION & LOCAL EXECUTION

```bash
git clone https://github.com/gangabhuvan/brain-tumor-classification-xai.git

cd brain-tumor-classification-xai

pip install -r requirements.txt

python app1.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

Tested with Python 3.10 and CUDA-enabled PyTorch.

---

# 📌 NOTES

* This repository is research-oriented, with most experimentation conducted through Jupyter notebooks in `model_experiments/`.

* The deployment system intentionally uses only the best-performing ConvNeXt + AO model for consistent inference performance.

* Model weights are automatically loaded/downloaded at runtime for simplified setup.

* The framework emphasizes reproducibility through unified preprocessing and consistent evaluation protocols.

---

# 📚 FUTURE WORK

* Inference optimization through quantization and lightweight deployment strategies
* GPU-enabled scalable cloud deployment using Docker/containerized infrastructure
* Clinical validation using real-world hospital datasets
* Multi-modal learning with additional imaging modalities and patient metadata
* Quantitative evaluation of explainability methods and clinician-centered XAI studies

---

# 🏁 CONCLUSION

This work presents a comprehensive and systematically evaluated deep learning framework for multi-class brain tumor classification. Across seven controlled experiments, the study demonstrates that optimization-driven training using the Aquila Optimizer significantly improves confidence calibration and prediction reliability while maintaining very high classification accuracy.

Beyond predictive performance, the framework emphasizes interpretability and practical deployment through integrated Explainable AI methods and a Flask-based diagnostic application. The results demonstrate the effectiveness of combining modern architectures, metaheuristic optimization, and clinically interpretable AI for robust medical imaging systems.

---

# 📢 PUBLICATION STATUS

A focused collaborative version of this research based on the EfficientNet-B0 + Aquila Optimizer framework has been accepted for Oral Presentation and Publication in the Springer Lecture Notes in Networks and Systems (LNNS) series at ICT4SD 2026.

## Accepted Paper

**Optimization-Driven Multi-Class Brain Tumor Classification Using EfficientNet-B0 and Aquila Optimizer**

* Conference: ICT4SD 2026, Goa, India
* Publisher: Springer LNNS Series
* Presentation Type: Oral Presentation

> Note:
> The accepted Springer conference paper focuses specifically on the EfficientNet-B0 + AO framework, while this repository additionally contains extended comparative experiments involving ConvNeXt, Swin Transformer, Random Search, K-Fold validation, Explainable AI (XAI), and deployment modules.

---

# 📚 CITATION

```bibtex
@inproceedings{patri2026efficientnet,
  title={Optimization-Driven Multi-Class Brain Tumor Classification Using EfficientNet-B0 and Aquila Optimizer},
  author={Patri, Bhuvankumar A. and Guptha, ChinmayKumar E. P. and Gowda, Rakshith B. and Arya, Charan K. and Biradar, Vidyadevi G.},
  booktitle={ICT4SD 2026},
  publisher={Springer LNNS},
  year={2026},
  note={Accepted for publication}
}
```
