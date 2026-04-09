import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import random

# ============================
# CONFIGURATION
# ============================
DATASET_DIR = "C:/Users/ganga/brain_tumor_dataset"
MODEL_PATH = "model/final_convnext_aq.pth"
BATCH_SIZE = 26
TRAIN_DIR = os.path.join(DATASET_DIR, "training")
TEST_DIR  = os.path.join(DATASET_DIR, "testing")

# ============================
# FIX SEEDS
# ============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ============================
# DEVICE
# ============================
device = torch.device("cpu")   # or "cuda" if available
torch.set_num_threads(os.cpu_count())
print(f"Using device: {device} with {torch.get_num_threads()} threads")

# ============================
# TRANSFORMS
# ============================
transform_eval = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ============================
# LOAD MODEL + FEATURE EXTRACTOR
# ============================
num_classes = 4
base_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
classifier_layer = base_model.classifier[2]

if isinstance(classifier_layer, nn.Sequential):
    linear_layer = next((m for m in classifier_layer if isinstance(m, nn.Linear)), None)
    in_features = linear_layer.in_features
else:
    in_features = classifier_layer.in_features

base_model.classifier[2] = nn.Sequential(
    nn.Dropout(p=0.2693893223768286),
    nn.Linear(in_features, num_classes)
)

model = base_model
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

class_names = ['Glioma','Meningioma','No Tumor','Pituitary']

# ============================
# FEATURE EXTRACTOR (Penultimate Layer)
# ============================
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool  # global pooling layer

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.mean([-2, -1])  # flatten (ConvNeXt uses mean pooling)
        return x

feature_extractor = FeatureExtractor(model).to(device)
feature_extractor.eval()

# ============================
# BUILD TRAIN FEATURE BANK
# ============================
def build_feature_bank(folder_path):
    print("\n🧠 Building feature bank from training set...")
    dataset = datasets.ImageFolder(root=folder_path, transform=transform_eval)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_feats = []
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Extracting Train Features"):
            inputs = inputs.to(device)
            feats = feature_extractor(inputs)
            feats = F.normalize(feats, dim=1)
            all_feats.append(feats.cpu())

    feature_bank = torch.cat(all_feats, dim=0)
    print(f"Feature bank size: {feature_bank.shape}")
    return feature_bank

train_feature_bank = build_feature_bank(TRAIN_DIR)

# ============================
# EVALUATION FUNCTION WITH SIMILARITY
# ============================
def evaluate_dataset(folder_path, label="test", feature_bank=None):
    print(f"\n🔍 Evaluating {label.upper()} dataset...")

    dataset = datasets.ImageFolder(root=folder_path, transform=transform_eval)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels, all_confs, all_sims = [], [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"{label.capitalize()} Set"):
            inputs, labels = inputs.to(device), labels.to(device)

            # --- Prediction ---
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)

            # --- Feature Similarity ---
            feats = feature_extractor(inputs)
            feats = F.normalize(feats, dim=1)  # normalize for cosine similarity

            # compute cosine similarity with entire train bank
            sim_matrix = torch.mm(feats.cpu(), feature_bank.T)
            max_sims, _ = sim_matrix.max(dim=1)
            all_sims.extend(max_sims.numpy())

            # collect
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

    # === Save CSV ===
    image_paths = [path[0] for path in dataset.samples]
    results_df = pd.DataFrame({
        "filename": [os.path.basename(p) for p in image_paths],
        "true_class": [class_names[i] for i in all_labels],
        "predicted_class": [class_names[i] for i in all_preds],
        "confidence": [round(float(c)*100,2) for c in all_confs],
        "similarity": [round(float(s)*100,2) for s in all_sims]
    })
    csv_path = f"results_{label}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✅ Saved predictions with similarity to {csv_path}")

    # === Metrics ===
    acc = accuracy_score(all_labels, all_preds)*100
    print(f"\n🎯 {label.upper()} Accuracy: {acc:.2f}%")
    print(f"\n📊 Classification Report ({label}):")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n🧩 Confusion Matrix ({label}):\n{cm}")

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{label.upper()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{label}.png")
    plt.close()
    print(f"🖼️ Saved confusion matrix as confusion_matrix_{label}.png")

    return acc

# ============================
# RUN TRAIN & TEST
# ============================
train_acc = evaluate_dataset(TRAIN_DIR, label="train", feature_bank=train_feature_bank)
test_acc  = evaluate_dataset(TEST_DIR, label="test", feature_bank=train_feature_bank)

print("\n✅ FINAL SUMMARY:")
print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy:  {test_acc:.2f}%")
print("\n📁 All results saved with similarity scores.")
