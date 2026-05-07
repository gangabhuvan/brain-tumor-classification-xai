# Aquila-Optimized Deep Learning for Brain Tumor Classification: A Comparative Study of Optimization Strategies, Model Stability, and Explainable AI

🎓 A focused collaborative version of this research based on the EfficientNet-B0 + Aquila Optimizer framework has been accepted for Oral Presentation and Publication at the :contentReference[oaicite:0]{index=0} in the Springer LNNS Series.

*A research-driven deep learning framework for brain tumor classification with metaheuristic optimization and clinically interpretable AI.*

---

## 👨‍💻 Primary Research & Implementation

**Bhuvankumar A. Patri**

---

## 🧠 Framework Highlights

- **Architectures:** ConvNeXt-Tiny, Swin Transformer-Tiny, EfficientNet-B0  
- **Optimization Strategies:** Aquila Optimizer (AO), Random Search (RS), Baseline Configuration  
- **Explainable AI (XAI):** Grad-CAM++, LIME, SHAP  
- **Deployment:** Flask-based diagnostic dashboard  

---

# 📌 ABSTRACT

Accurate and early diagnosis of brain tumors is crucial for effective clinical intervention and improved patient survival rates. Although Magnetic Resonance Imaging (MRI) is widely used for tumor detection, manual interpretation is time-consuming and subject to inter-observer variability.

This work presents a deep learning framework for multi-class brain tumor classification using a unified preprocessing pipeline and systematic experimental evaluation.

All experiments were conducted on the Kaggle Brain Tumor MRI dataset consisting of 7,023 T1-weighted contrast-enhanced MRI images across four classes:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

A total of seven controlled experiments were performed across three modern architectures—ConvNeXt, Swin Transformer, and EfficientNet-B0—combined with Aquila Optimizer (AO), Random Search (RS), and baseline configurations.

All models achieved consistently high performance (>99% accuracy), indicating strong feature separability in MRI-based tumor classification, with observable differences in confidence calibration measured using mean Average Precision (mAP).

Among all configurations, **ConvNeXt + AO** achieved the best overall performance with:

- **Accuracy:** 99.69%
- **MCC:** 0.9959
- **Maximum observed mAP:** 1.0000

These results indicate superior ranking confidence and improved decision-boundary refinement.

Additional configurations, including Swin Transformer (RS and AO) and EfficientNet-B0 + AO, also demonstrated highly competitive performance, validating the robustness of transformer-based and CNN-based architectures for medical imaging tasks.

To improve interpretability and clinical trustworthiness, a multi-level Explainable AI (XAI) framework integrating Grad-CAM++, LIME, and SHAP was applied exclusively to the best-performing ConvNeXt + AO model.

A Flask-based diagnostic dashboard was further developed to demonstrate real-time prediction and explainability capabilities.

---

# 🚀 KEY CONTRIBUTIONS

- Controlled comparative analysis of ConvNeXt, Swin Transformer, and EfficientNet-B0  
- Hyperparameter optimization using Aquila Optimizer (AO)  
- Comparative validation against Random Search and baseline configurations  
- Achievement of a maximum observed mAP of **1.0000**  
- Integration of multi-level Explainable AI (Grad-CAM++, LIME, SHAP)  
- Development of an end-to-end Flask-based diagnostic application  
- Unified preprocessing and evaluation pipeline for reproducibility  
- Comparative evaluation of optimization strategies for medical image classification  

---

# 🔬 METHODOLOGY

The proposed framework follows a consistent experimental pipeline across all seven experiments to ensure fair comparison between architectures and optimization strategies.

---

## 1️⃣ Dataset Preparation

- Dataset loaded using `ImageFolder` from the Kaggle Brain Tumor MRI dataset  
- Separate `Training/` and `Testing/` directories utilized  
- Class labels automatically extracted from the folder structure  

---

## 2️⃣ Data Preprocessing & Augmentation

A unified preprocessing pipeline was applied across all models.

### Training Transformations

- Resize to `224 × 224`
- Random Horizontal Flip
- Random Rotation (`±20°`)
- Color Jitter (brightness and contrast)
- Normalization using ImageNet mean and standard deviation

### Validation/Test Transformations

- Resize to `224 × 224`
- Normalization only (without augmentation)

---

## 3️⃣ Data Splitting Strategy

Two evaluation strategies were implemented.

### Stratified Single Split

- 80% Training
- 20% Validation
- Stratified using `train_test_split` to preserve class balance

### 5-Fold Cross-Validation

- Dataset divided into 5 folds using `KFold`
- Each fold used once as validation data
- Final performance averaged across folds

---

## 4️⃣ Model Architectures

Three architectures were evaluated:

- **ConvNeXt-Tiny** (ImageNet pretrained)
- **Swin Transformer-Tiny**
- **EfficientNet-B0**

For all models:

- Final classification layer modified for 4-class classification
- Dropout regularization added/tuned

---

## 5️⃣ Training Configuration

- **Loss Function:** `CrossEntropyLoss`
- **Optimizer:** `AdamW`
- **Scheduler:** `CosineAnnealingLR`
- **Mixed Precision Training:** `torch.cuda.amp`
- **Batch Size:** ~26–32 (experiment-dependent)
- **Epochs:** Up to 20
- **Early Stopping:** Enabled to prevent overfitting

---

## 6️⃣ Hyperparameter Optimization

Three optimization strategies were evaluated:

- **Baseline:** Fixed hyperparameters
- **Random Search (RS):** Random parameter sampling
- **Aquila Optimizer (AO):** Metaheuristic optimization

### AO Optimization Targets

- Learning Rate
- Dropout Probability
- Batch Size

### Example (ConvNeXt + AO)

| Parameter | Value |
|---|---|
| Learning Rate | ~1.33e-4 |
| Dropout | ~0.27 |
| Batch Size | ~26 |

---

## 7️⃣ Evaluation Metrics

Performance was evaluated using:

- **Accuracy** → Overall classification correctness  
- **MCC (Matthews Correlation Coefficient)** → Balanced predictive reliability  
- **mAP (mean Average Precision)** → Confidence calibration and ranking quality  
- **Confusion Matrix**  
- **Classification Report**  
- **Class-wise Recall** (critical in medical diagnosis)

---

## 8️⃣ Inference & Deployment

- Best-performing model (**ConvNeXt + AO**) selected
- Integrated into a Flask-based web application
- Real-time prediction with XAI visualizations

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
```

---

# 📊 EXPERIMENTAL RESULTS

**Test Samples (N = 1,311)**

| Exp | Model Configuration | Optimization | Accuracy (%) | MCC | mAP | Errors |
|---|---|---|---|---|---|---|
| 01 | ConvNeXt (Single Split) | AO | **99.69** | **0.9959** | **1.0000** | 4 |
| 02 | ConvNeXt (Single Split) | Baseline | 99.69 | 0.9959 | 0.9999 | 4 |
| 03 | Swin Transformer | RS | 99.62 | 0.9949 | 0.9999 | 5 |
| 04 | Swin Transformer | AO | 99.62 | 0.9949 | 0.9995 | 5 |
| 05 | ConvNeXt (5-Fold CV) | AO | 99.47 | 0.9928 | 0.9998 | 7 |
| 06 | EfficientNet-B0 | AO | 99.47 | 0.9929 | 0.9999 | 7 |
| 07 | ConvNeXt (5-Fold CV) | Baseline | 99.24 | 0.9898 | 0.9997 | 10 |

---

# 🔍 KEY OBSERVATIONS

- **ConvNeXt + AO** and **Swin Transformer (RS & AO)** achieved **100% recall for the Meningioma class**, indicating no missed meningioma cases.

- **EfficientNet-B0 + AO** achieved **100% recall for the No Tumor class**, demonstrating strong reliability in healthy-case detection.

- Although overall accuracy remained similarly high across models, class-wise behavior differed, emphasizing the importance of per-class medical evaluation.

- ConvNeXt + AO demonstrated stronger tumor-detection behavior, while EfficientNet-B0 showed stronger reliability for normal-case classification.

- ROC-AUC values approached saturation (~1.0) across all models, making **mAP** and **MCC** more informative metrics for confidence calibration and prediction reliability.

---

# 💡 WHY RECALL MATTERS

In medical diagnosis, recall is a critical metric because false negatives may result in missed tumor cases. High recall ensures safer clinical decision-support behavior and reduces the probability of overlooking clinically significant abnormalities.

---

# 🧠 EXPLAINABLE AI (XAI)

Applied exclusively to the best-performing **ConvNeXt + AO** model.

| Method | Purpose |
|---|---|
| Grad-CAM++ | Tumor localization |
| LIME | Local feature importance |
| SHAP | Pixel-level contribution analysis |

These methods improve transparency, interpretability, and clinical trustworthiness.

---

# 🌐 WEB APPLICATION

The Flask-based diagnostic dashboard includes:

- Secure login system
- MRI upload interface
- Real-time prediction
- Explainability visualization
- Clinical-style feedback
- PDF report generation

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

## Clone Repository

```bash
git clone https://github.com/gangabhuvan/brain-tumor-classification-xai.git
```

## Navigate to Project

```bash
cd brain-tumor-classification-xai
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python app1.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

Tested using:

- Python 3.10
- CUDA-enabled PyTorch

---

# 📌 NOTES

- This repository is primarily research-oriented, with most experimentation conducted through Jupyter notebooks under `model_experiments/`.

- The deployment system intentionally utilizes only the best-performing ConvNeXt + AO model for consistent inference performance.

- Model weights are automatically loaded/downloaded at runtime for simplified setup.

- The framework emphasizes reproducibility through unified preprocessing and consistent experimental protocols.

---

# 📚 FUTURE WORK

- Model quantization and lightweight inference optimization  
- GPU-enabled scalable cloud deployment using Docker/containerized infrastructure  
- Clinical validation using real-world hospital datasets  
- Multi-modal learning with additional imaging modalities and patient metadata  
- Quantitative evaluation of XAI methods through clinician-centered studies  

---

# 🏁 CONCLUSION

This work presents a comprehensive and systematically evaluated deep learning framework for multi-class brain tumor classification.

Across seven controlled experiments, the study demonstrates that optimization-driven training using the Aquila Optimizer significantly improves confidence calibration and prediction reliability while maintaining extremely high classification accuracy.

Beyond predictive performance, the framework emphasizes interpretability and practical deployment through integrated Explainable AI methods and a Flask-based diagnostic application.

The results demonstrate the effectiveness of combining modern deep learning architectures, metaheuristic optimization, and clinically interpretable AI for robust medical imaging systems.

---

# 📢 PUBLICATION STATUS

A focused collaborative version of this research based on the EfficientNet-B0 + Aquila Optimizer framework has been accepted for Oral Presentation and Publication in the Springer Lecture Notes in Networks and Systems (LNNS) Series at the :contentReference[oaicite:1]{index=1}.

## Accepted Paper

### **Optimization-Driven Multi-Class Brain Tumor Classification Using EfficientNet-B0 and Aquila Optimizer**

- **Conference:** ICT4SD 2026, Goa, India  
- **Publisher:** Springer LNNS Series  
- **Presentation Type:** Oral Presentation  

> **Note:**  
> The accepted Springer conference paper specifically focuses on the EfficientNet-B0 + AO framework, whereas this repository additionally includes extended comparative experiments involving ConvNeXt, Swin Transformer, Random Search, K-Fold validation, Explainable AI (XAI), and deployment modules.

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

---

# ⭐ ACKNOWLEDGEMENT

If you find this project useful for research or academic purposes, consider starring the repository and citing the work.
