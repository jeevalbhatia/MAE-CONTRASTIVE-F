import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import models_mae

checkpoint_folder = r"C:\Users\Jeeval Bhatia\mae-contrastive\test_can_output"
data_root = r"C:\Users\Jeeval Bhatia\Downloads\archive (3)\chest_xray"
device = "cpu"
model_name = "mae_vit_base_patch16"
norm_pix_loss = False
noise_loss = False

train_dir = os.path.join(data_root, "train")
val_dir = os.path.join(data_root, "val")
test_dir = os.path.join(data_root, "test")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

ckpt_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(".pth")]
if not ckpt_files:
    raise FileNotFoundError(f"No .pth files found in {checkpoint_folder}")
ckpt_files.sort()
checkpoint_path = os.path.join(checkpoint_folder, ckpt_files[-1])
print(f"Loading checkpoint: {checkpoint_path}")

model = models_mae.__dict__[model_name](norm_pix_loss=norm_pix_loss, noise_loss=noise_loss)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

def extract_features(folder):
    features = []
    labels = []
    class_names = sorted(os.listdir(folder))
    total_images = sum(len(os.listdir(os.path.join(folder, c))) for c in class_names if os.path.isdir(os.path.join(folder, c)))

    with tqdm(total=total_images, desc=f"Extracting from {os.path.basename(folder)}", ncols=100) as pbar:
        for cls_idx, cls in enumerate(class_names):
            cls_folder = os.path.join(folder, cls)
            if not os.path.isdir(cls_folder):
                continue
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model.forward_features(img_tensor)
                    if isinstance(out, tuple):
                        out = out[0]
                    if out.dim() == 1:
                        out = out.unsqueeze(0)
                features.append(out.squeeze(0).cpu().numpy())
                labels.append(cls_idx)
                pbar.update(1)
    return np.array(features), np.array(labels), class_names

print("Extracting train features...")
X_train, y_train, class_names = extract_features(train_dir)
print("Extracting val features...")
X_val, y_val, _ = extract_features(val_dir)
print("Extracting test features...")
X_test, y_test, _ = extract_features(test_dir)

print("Training linear classifier...")
clf = LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc*100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))
