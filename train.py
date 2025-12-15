import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

from models.gramnet import GramNet
from data.dataset import DeepfakeCSVDataset

# ================= CONFIG =================
DATA_ROOT = "data"
TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"

BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

# ---------------- Transforms ----------------
def train():
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3)], p=0.3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    # ---------------- Datasets ----------------
    train_dataset = DeepfakeCSVDataset(
        root=DATA_ROOT,
        csv_file=TRAIN_CSV,
        transform=train_tfms
    )

    val_dataset = DeepfakeCSVDataset(
        root=DATA_ROOT,
        csv_file=VAL_CSV,
        transform=val_tfms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # ---------------- Model ----------------
    model = GramNet(num_classes=2).to(DEVICE)

    # GRAM_INIT_PATH = "graminit.pth"
    # if os.path.exists(GRAM_INIT_PATH):
    #     print("Loading Gram initialization weights...")
    #     state_dict = torch.load(GRAM_INIT_PATH, map_location="cpu")
    #     missing, unexpected = model.load_state_dict(
    #         state_dict, strict=False
    #     )
    #     print("Missing keys:", missing)
    #     print("Unexpected keys:", unexpected)
    # else:
    #     print("No graminit.pth found, training from scratch.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    # ================= TRAINING LOOP =================
    for epoch in range(EPOCHS):
        # -------- Train --------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in loop:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

            loop.set_postfix(
                loss=train_loss / train_total,
                acc=100. * train_correct / train_total
            )

        train_acc = 100. * train_correct / train_total
        train_loss /= train_total

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="[Val]"):
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100. * val_correct / val_total
        val_loss /= val_total

        print(
            f"\nEpoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%"
        )

        # -------- Save best --------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                "checkpoints/gramnet_best.pth"
            )
            print(f"Saved new best model (Val Acc: {val_acc:.2f}%)")

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')  # optional, explicit
    train()