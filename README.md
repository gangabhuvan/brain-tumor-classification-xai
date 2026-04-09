# 🧠 Deep Learning Models for Brain Tumor Classification with Explainable AI

*Author:* Bhuvankumar A Patri  
*Architecture:* ConvNeXt, Swin Transformer, EfficientNet-B0  
*Optimization:* Aquila Optimizer (AQ), Random Search (RS), Baseline  
*Explainability:* Grad-CAM++, LIME, SHAP  

---

## 📌 ABSTRACT

Accurate and early diagnosis of brain tumors is crucial for effective clinical intervention and improved patient survival rates. Although Magnetic Resonance Imaging (MRI) is widely used for detection, manual interpretation is time-consuming and subject to inter-observer variability. This work presents a deep learning framework for multi-class brain tumor classification using a unified preprocessing pipeline and systematic experimental evaluation. A total of seven controlled experiments are conducted across three modern architectures—ConvNeXt, Swin Transformer, and EfficientNet-B0—combined with Aquila Optimizer (AQ), Random Search (RS), and baseline configurations. All models achieve high classification performance exceeding 99% accuracy, with observable differences in confidence calibration measured using mean Average Precision (mAP). Among all configurations, ConvNeXt optimized with AQ achieves the best performance with 99.69% accuracy, MCC of 0.9959, and a perfect mAP of 1.0000, indicating superior ranking confidence and decision boundary refinement. Other configurations, including Swin Transformer (RS and AQ) and EfficientNet-B0 (AQ), demonstrate competitive performance, validating architectural robustness. A multi-level Explainable AI (XAI) framework integrating Grad-CAM++, LIME, and SHAP is applied exclusively to the best-performing ConvNeXt + AQ model to ensure interpretability. A Flask-based diagnostic dashboard is developed to demonstrate real-time prediction and explainability capabilities.

---

## 🚀 KEY CONTRIBUTIONS

- Controlled comparison of ConvNeXt, Swin Transformer, EfficientNet-B0  
- Metaheuristic optimization using Aquila Optimizer (AQ)  
- Validation against Random Search and baseline models  
- Achievement of *perfect mAP (1.0000)*  
- Multi-level XAI (Grad-CAM++, LIME, SHAP)  
- End-to-end Flask-based diagnostic system  

---

## 📊 EXPERIMENTAL RESULTS (N = 1,311)

| Exp | Model Configuration| Optimizer| Accuracy (%) | MCC     | mAP     | Errors |
|-----|--------------------|----------|--------------|---------|---------|--------|
| 01  | ConvNeXt (Single)  | Aquila   | *99.69*      | *0.9959*| *1.0000*|   *4*  |
| 02  | ConvNeXt (Single)  | Baseline | 99.69        | 0.9959  |  0.9999 |    4   |
| 03  | Swin Transformer   | RS       | 99.62        | 0.9949  |  0.9999 |    5   |
| 04  | Swin Transformer   | Aquila   | 99.62        | 0.9949  |  0.9995 |    5   |
| 05  | ConvNeXt (K-Fold)  | Aquila   | 99.47        | 0.9928  |  0.9998 |    7   |
| 06  | EfficientNet-B0    | Aquila   | 99.47        | 0.9929  |  0.9999 |    7   |
| 07  | ConvNeXt (K-Fold)  | Baseline | 99.24        | 0.9898  |  0.9997 |    10  |

---

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
├── static/                         # Frontend assets
│   ├── css/
│   ├── js/
│   ├── background.jpg
│   ├── confusion_matrix_train.png
│   ├── confusion_matrix_test.png
│   ├── favicon.ico
│   ├── nmit_logo.jpeg
│   └── outputs/
│
├── templates/                      # HTML templates
│   ├── base.html
│   ├── index1.html
│   ├── login.html
│   └── register.html
│
├── uploads/                        # Uploaded MRI images
│
├── utils/                          # Core logic
│   └── explainer1.py
│
├── app1.py                         # Flask backend
├── evaluate_dataset.py            # Evaluation script
├── generate_train_features.py     # Feature extraction
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

## ⚙️ INSTALLATION

bash
git clone https://github.com/gangabhuvan/brain-tumor-classification-xai.git
cd brain-tumor-classification-xai
pip install -r requirements.txt
python app1.py


Open:


http://127.0.0.1:5000


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
