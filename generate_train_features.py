import os
import torch
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
import torch.nn.functional as F

# ------------------------------
# Settings
# ------------------------------
train_folder = "dataset/train"  # training images folder
save_path = "train_features.pt"
batch_size = 26
pth_path = "model/final_convnext_aq.pth"  # your trained model weights

# Use all CPU threads
torch.set_num_threads(os.cpu_count())
device = torch.device("cpu")  # CPU-only

# ------------------------------
# Load model
# ------------------------------
model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
num_classes = 4
classifier_layer = model.classifier[2]
if isinstance(classifier_layer, torch.nn.Sequential):
    linear_layer = next((m for m in classifier_layer if isinstance(m, torch.nn.Linear)), None)
    in_features = linear_layer.in_features
else:
    in_features = classifier_layer.in_features

# Replace classifier
model.classifier[2] = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2693893223768286),
    torch.nn.Linear(in_features, num_classes)
)

# Load your trained weights
model.load_state_dict(torch.load(pth_path, map_location=device))
model.to(device).eval()

# ------------------------------
# Image transform
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------------------------
# Collect all image paths
# ------------------------------
img_paths = []
for cls in os.listdir(train_folder):
    cls_folder = os.path.join(train_folder, cls)
    if not os.path.isdir(cls_folder):
        continue
    for fname in os.listdir(cls_folder):
        fpath = os.path.join(cls_folder, fname)
        img_paths.append(fpath)

print(f"Total images: {len(img_paths)}")

# ------------------------------
# Precompute features in batches
# ------------------------------
features_list = []
count = 0

for i in range(0, len(img_paths), batch_size):
    batch_paths = img_paths[i:i+batch_size]
    batch_tensors = []
    for path in batch_paths:
        img = Image.open(path).convert("RGB")
        batch_tensors.append(transform(img))
    batch = torch.stack(batch_tensors).to(device)

    with torch.no_grad():
        feats = model.features(batch)
        feats = feats.flatten(1)
        feats = F.normalize(feats, dim=1)
        features_list.append(feats)

    count += len(batch_paths)
    print(f"Processed {count}/{len(img_paths)} images...")

# ------------------------------
# Save all features
# ------------------------------
train_features = torch.cat(features_list, dim=0)
torch.save(train_features, save_path)
print(f"Saved train features to {save_path}")



