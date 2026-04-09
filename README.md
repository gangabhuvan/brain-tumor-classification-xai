🧠 Deep Learning Models for Brain Tumor Classification with Metaheuristic Optimization and Explainable AI

Author: Bhuvankumar A Patri
Architecture: ConvNeXt, Swin Transformer, EfficientNet-B0
Optimization: Aquila Optimizer (AQ), Random Search (RS), Baseline
Explainability: Grad-CAM++, LIME, SHAP

📄 ABSTRACT

Accurate and early diagnosis of brain tumors is critical for effective clinical intervention and improved patient outcomes. Although Magnetic Resonance Imaging (MRI) is widely used for detection, manual interpretation is time-consuming and subject to inter-observer variability. This work presents a deep learning framework for multi-class brain tumor classification using a unified preprocessing pipeline and systematic experimental evaluation. A total of seven controlled experiments are conducted across three modern architectures—ConvNeXt-Tiny, Swin Transformer, and EfficientNet-B0—combined with Aquila Optimizer (AQ), Random Search (RS), and baseline configurations. The results consistently demonstrate high classification performance exceeding 99% accuracy across models, with observable differences in confidence calibration measured using mean Average Precision (mAP). Among all configurations, ConvNeXt-Tiny with AQ achieves the best performance with 99.69% accuracy, MCC of 0.9959, and mAP of 1.0000. A multi-level Explainable AI (XAI) framework integrating Grad-CAM++, LIME, and SHAP is applied to the best-performing model to enhance interpretability. A Flask-based diagnostic dashboard is developed to demonstrate real-time prediction and explainability capabilities. These findings indicate that metaheuristic optimization improves model reliability and that explainability plays a critical role in clinical decision support.

🚀 KEY CONTRIBUTIONS
	•	Comparative evaluation of ConvNeXt, Swin Transformer, and EfficientNet-B0 under a unified pipeline
	•	Study of Aquila Optimizer (AQ), Random Search (RS), and baseline tuning strategies
	•	Analysis of confidence calibration using mAP alongside accuracy and MCC
	•	Identification of ConvNeXt + AQ as the best-performing configuration
	•	Integration of multi-level explainability (Grad-CAM++, LIME, SHAP)
	•	Development of a Flask-based diagnostic dashboard for real-time inference and visualization

📊 EXPERIMENTAL RESULTS (N = 1,311)

Exp	Model Configuration	 Optimization	         Accuracy (%)	  MCC	          mAP	    Errors
01	ConvNeXt (Single)	    Aquila (AQ)	           99.69	     0.9959	      1.0000	    4
02	ConvNeXt (Single)	    Baseline	             99.69	     0.9959	      0.9999	    4
03	Swin Transformer	    Random Search (RS)	   99.62	     0.9949       0.9999	    5
04	Swin Transformer	    Aquila (AQ)	           99.62	     0.9949	      0.9995	    5
05	ConvNeXt (K-Fold)	    Aquila (AQ)	           99.47	     0.9928	      0.9998	    7
06	EfficientNet-B0	      Aquila (AQ)	           99.47	     0.9929	      0.9999	    7
07	ConvNeXt (K-Fold)	    Baseline	             99.24	     0.9898	      0.9997	    10

🧠 MODEL INSIGHTS
	•	ConvNeXt demonstrates strong performance for MRI-based classification tasks
	•	Aquila Optimizer improves confidence calibration compared to baseline configurations
	•	Swin Transformer provides competitive results with attention-based modeling
	•	EfficientNet-B0 maintains high accuracy with slightly higher error rates
	•	K-Fold validation confirms model stability across different data splits

🔍 EXPLAINABLE AI (XAI)

Applied to the best-performing model (ConvNeXt + AQ)
	•	Grad-CAM++ → Localization of tumor regions
	•	LIME → Identification of influential superpixels
	•	SHAP → Pixel-level contribution analysis

This multi-level approach enhances interpretability and supports transparent decision-making.

🖥️ FLASK DIAGNOSTIC DASHBOARD

The project includes a web-based interface for practical usage:
	•	MRI image upload
	•	Tumor classification prediction
	•	Confidence score display
	•	Visualization of Grad-CAM++, LIME, and SHAP outputs
	•	Explainability metrics (IoU, Dice, TEAS, robustness)
	•	Automated feedback generation
	•	Downloadable PDF diagnostic report

📁 PROJECT STRUCTURE

brain-tumor-classification-xai/
│
├── model_experiments/
├── static/
├── templates/
├── utils/
├── model/
├── app1.py
├── results_train.csv
├── results_test.csv
├── requirements.txt
└── README.md

⚙️ INSTALLATION & LOCAL EXECUTION

1. Clone Repository

git clone https://github.com/gangabhuvan/brain-tumor-classification-xai.git
cd brain-tumor-classification-xai

2. Install Dependencies

pip install -r requirements.txt

3. Run Application

python app1.py

4. Open in Browser

http://127.0.0.1:5000

🔐 DEFAULT LOGIN

Username: admin  
Password: admin123

⚠️ NOTES
	•	Model weights are not stored in the repository due to size limitations
	•	They are automatically downloaded from Google Drive during runtime
	•	Ensure an active internet connection for the first run
	•	All required CSV files for evaluation are included in the repository

📈 CONCLUSION

This project demonstrates that modern deep learning architectures combined with metaheuristic optimization can achieve highly reliable performance in brain tumor classification. The integration of explainable AI further improves transparency, making the system suitable for decision-support scenarios.

📌 PROJECT STATUS

This repository contains the implementation and experimental results of an ongoing research study in deep learning and explainable AI for medical imaging.

👨‍💻 AUTHOR

Bhuvankumar A Patri
B.E. Information Science & Engineering
Nitte Meenakshi Institute of Technology, Bengaluru
