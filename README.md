# 🧠 Deep Learning Models for Brain Tumor Classification with Explainable AI

**Author:** Bhuvankumar A Patri  
**Architecture:** ConvNeXt, Swin Transformer, EfficientNet-B0  
**Optimization:** Aquila Optimizer (AQ), Random Search (RS), Baseline  
**Explainability:** Grad-CAM++, LIME, SHAP  

---

## 📄 ABSTRACT

Accurate and early diagnosis of brain tumors is critical for effective clinical intervention and improved patient outcomes. Although Magnetic Resonance Imaging (MRI) is widely used for tumor detection, manual interpretation is time-consuming and subject to inter-observer variability. This work presents a deep learning framework for multi-class brain tumor classification using a unified preprocessing pipeline and systematic experimental evaluation. A total of seven controlled experiments are conducted across three modern architectures—ConvNeXt-Tiny, Swin Transformer, and EfficientNet-B0—combined with Aquila Optimizer (AQ), Random Search (RS), and baseline configurations. The results consistently demonstrate high classification performance exceeding 99% accuracy across all models, with observable differences in confidence calibration measured using mean Average Precision (mAP). Among all configurations, ConvNeXt-Tiny with AQ achieves the best performance with 99.69% accuracy, MCC of 0.9959, and a perfect mAP of 1.0000, indicating superior confidence reliability. A multi-level explainable AI (XAI) framework integrating Grad-CAM++, LIME, and SHAP is applied exclusively to the best-performing model to enhance interpretability. Additionally, a Flask-based diagnostic dashboard is developed to demonstrate real-time prediction and explainability capabilities. These findings highlight the effectiveness of metaheuristic optimization in improving model reliability and the importance of explainability in clinical decision support systems.

---

## 🚀 KEY CONTRIBUTIONS

- Comparative evaluation of ConvNeXt, Swin Transformer, and EfficientNet-B0 under a unified experimental pipeline  
- Analysis of Aquila Optimizer (AQ), Random Search (RS), and baseline tuning strategies  
- Identification of confidence differences using mAP beyond standard accuracy metrics  
- Integration of multi-level explainable AI (Grad-CAM++, LIME, SHAP) on the best-performing model  
- Development of a Flask-based diagnostic dashboard for real-time prediction and interpretability  

---

## 📊 EXPERIMENTAL RESULTS

Seven controlled experiments were conducted to evaluate the impact of architecture and optimization strategies.

| Exp | Model Configuration      | Optimizer        | Accuracy (%) | MCC    | mAP    | Errors |
|-----|--------------------------|------------------|--------------|--------|--------|--------|
| 01  | ConvNeXt (Single)        | Aquila (AQ)      | 99.69        | 0.9959 | 1.0000 | 4      |
| 02  | ConvNeXt (Single)        | Baseline         | 99.69        | 0.9959 | 0.9999 | 4      |
| 03  | Swin Transformer         | Random Search    | 99.62        | 0.9949 | 0.9999 | 5      |
| 04  | Swin Transformer         | Aquila (AQ)      | 99.62        | 0.9949 | 0.9995 | 5      |
| 05  | ConvNeXt (K-Fold)        | Aquila (AQ)      | 99.47        | 0.9928 | 0.9998 | 7      |
| 06  | EfficientNet-B0          | Aquila (AQ)      | 99.47        | 0.9929 | 0.9999 | 7      |
| 07  | ConvNeXt (K-Fold)        | Baseline         | 99.24        | 0.9898 | 0.9997 | 10     |

**Insight:** ConvNeXt + AQ achieves perfect mAP (1.0000), indicating superior confidence calibration and ranking consistency.

---

## 🔬 EXPLAINABLE AI FRAMEWORK

Explainability is applied **only to the best-performing model (ConvNeXt + AQ)**:

- **Grad-CAM++** → Localizes tumor regions  
- **LIME** → Highlights influential superpixels  
- **SHAP** → Provides pixel-level contribution analysis  

This multi-method approach ensures interpretability consistency and clinical trust.

---

## 🖥️ WEB APPLICATION (FLASK DASHBOARD)

A diagnostic web application is developed for real-time usage.

### Features:
- MRI image upload  
- Tumor classification with confidence score  
- Similarity analysis  
- Grad-CAM++, LIME, SHAP visualizations  
- Radiologist-style feedback summary  
- Explainability grading system  
- Downloadable PDF report  

---

## 📁 PROJECT STRUCTURE

```bash
├── model_experiments/
├── static/
├── templates/
├── utils/
├── app1.py
├── evaluate_dataset.py
├── results_train.csv
├── results_test.csv
└── requirements.txt
```

---

## ⚙️ INSTALLATION & RUN

```bash
git clone https://github.com/gangabhuvan/brain-tumor-classification-xai.git
cd brain-tumor-classification-xai
pip install -r requirements.txt
python app1.py
```

Open browser:
```
http://127.0.0.1:5000
```

---

## 📌 NOTES

- Model weights are automatically downloaded from Google Drive (.pth file)
- CSV files are included for metrics computation  
- Designed for local execution  

---

## 📜 LICENSE

This project is for academic and research purposes only.

---

## 👨‍💻 AUTHOR

**Bhuvankumar A Patri**
