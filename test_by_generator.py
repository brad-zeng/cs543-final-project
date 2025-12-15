import torch
from models.gramnet import GramNet
from torchvision import transforms
from data.dataset import TestDataset, TestDatasetWithGenerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score
)
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


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

DATA_DIR = "data"
TEST_SPLIT_DIR = "test.csv"
test_dataset = TestDatasetWithGenerator(
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
gen_labels = defaultdict(list)
gen_probs = defaultdict(list)
gen_preds = defaultdict(list)

loop = tqdm(test_loader)

with torch.no_grad():
    for imgs, labels, generators in tqdm(test_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, 0]
        preds = torch.argmax(outputs, dim=1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        for g, y, p, pr in zip(
            generators,
            labels.cpu().numpy(),
            preds.cpu().numpy(),
            probs.cpu().numpy(),
        ):
            gen_labels[g].append(y)
            gen_preds[g].append(p)
            gen_probs[g].append(pr)

# with torch.no_grad():
#     for imgs, labels in loop:
#         imgs = imgs.to(device)
#         labels = labels.to(device)

#         outputs = model(imgs)               # (B, 2)
#         probs = torch.softmax(outputs, dim=1)[:, 1]  # P(fake)
#         preds = torch.argmax(outputs, dim=1)

#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

#         all_labels.extend(labels.cpu().numpy())
#         all_preds.extend(preds.cpu().numpy())
#         all_probs.extend(probs.cpu().numpy())

real_indices = [i for i, y in enumerate(all_labels) if y == 1]
real_labels = [all_labels[i] for i in real_indices]
real_preds  = [all_preds[i] for i in real_indices]
real_probs  = [all_probs[i] for i in real_indices]

for gen in sorted(gen_labels.keys()):
    if gen.lower() == "real":
        continue  # skip real itself

    # generator images
    y_true_gen = gen_labels[gen]    # all 0
    y_pred_gen = gen_preds[gen]
    y_prob_gen = gen_probs[gen]

    # combine with real images
    y_true = y_true_gen + real_labels
    y_pred = y_pred_gen + real_preds
    y_prob = y_prob_gen + real_probs

    pr_auc = average_precision_score(y_true, y_prob)
    acc_gen = accuracy_score(y_true, y_pred)

    print(
        f"{gen:15s} | PR-AUC: {pr_auc:.3f} | "
        f"N: {len(y_true)} | Accuracy Score: {acc_gen:.3f}"
    )
    
# real_labels = [1 for y in all_labels if y == 1]
# real_probs = [p for y, p in zip(all_labels, all_probs) if y == 1]


# all_labels = np.array(all_labels)
# all_preds = np.array(all_preds)
# all_probs = np.array(all_probs)


# print(f"Test Accuracy: {correct / total:.4f}")
# roc_auc = roc_auc_score(all_labels, all_probs)
# print(f"ROC-AUC: {roc_auc:.4f}")
# ap = average_precision_score(all_labels, all_probs)
# print(f"PR-AUC: {ap:.4f}")
# p, r, f1, _ = precision_recall_fscore_support(
#     all_labels, all_preds, average="binary"
# )
# print(f"Precision: {p:.4f}")
# print(f"Recall:    {r:.4f}")
# print(f"F1-score:  {f1:.4f}")
# bal_acc = balanced_accuracy_score(all_labels, all_preds)
# print(f"Balanced Accuracy: {bal_acc:.4f}")
# cm = confusion_matrix(all_labels, all_preds)
# print("Confusion Matrix:")
# print(cm)
# fpr, tpr, _ = roc_curve(all_labels, all_probs)
# fnr = 1 - tpr
# eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
# print(f"EER: {eer:.4f}")

# #plot roc curve
# plt.figure()
# plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
# plt.plot([0, 1], [0, 1], linestyle="--")  # random baseline
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()

# #plot pr curve
# precision, recall, _ = precision_recall_curve(all_labels, all_probs)
# plt.figure()
# plt.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision–Recall Curve")
# plt.legend()
# plt.grid(True)
# plt.show()