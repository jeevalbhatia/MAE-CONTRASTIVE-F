import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Paths
data_root = r"C:\Users\Jeeval Bhatia\Downloads\archive (3)\chest_xray"
train_dir = os.path.join(data_root, "train")
val_dir = os.path.join(data_root, "val")
test_dir = os.path.join(data_root, "test")

# Transform: resize to 224x224 and convert to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# DataLoaders
batch_size = 2  # adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"✅ Datasets loaded:")
print(f"  Train: {len(train_dataset)} images")
print(f"  Val: {len(val_dataset)} images")
print(f"  Test: {len(test_dataset)} images")
