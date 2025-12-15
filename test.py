import torch
from models.gramnet import GramNet
from torchvision import transforms
from data.dataset import TestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GramNet(num_classes=2)
model.load_state_dict(torch.load("checkpoints/gramnet_best.pth", map_location=device))
model.to(device)
model.eval()

test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

#test using deepdetect2025 dataset
DATA_DIR = "ddata/test"
TEST_SPLIT_DIR = "big_test.csv"
test_dataset = TestDataset(
    csv_file=TEST_SPLIT_DIR,
    root_dir=DATA_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=True
)

correct = 0
total = 0

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    loop = tqdm(test_loader)
    for imgs, labels in loop:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)               # (B, 2)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # P(fake)
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

print(f"Test Accuracy: {correct / total:.4f}")
roc_auc = roc_auc_score(all_labels, all_probs)
print(f"ROC-AUC: {roc_auc:.4f}")
ap = average_precision_score(all_labels, all_probs)
print(f"PR-AUC: {ap:.4f}")
p, r, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="binary"
)
print(f"Precision: {p:.4f}")
print(f"Recall:    {r:.4f}")
print(f"F1-score:  {f1:.4f}")
bal_acc = balanced_accuracy_score(all_labels, all_preds)
print(f"Balanced Accuracy: {bal_acc:.4f}")
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)
fpr, tpr, _ = roc_curve(all_labels, all_probs)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
print(f"EER: {eer:.4f}")

#plot roc curve
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")  # random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#plot pr curve
precision, recall, _ = precision_recall_curve(all_labels, all_probs)
plt.figure()
plt.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.grid(True)
plt.show()