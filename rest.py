import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

torch.backends.cudnn.benchmark = True


# ======================================
# Encoder (same architecture used in SimCLR)
# ======================================

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128*28*28,256),
            nn.ReLU()
        )

    def forward(self,x):
        return self.features(x)


# ======================================
# MAIN
# ======================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ======================================
    # Transforms
    # ======================================

    train_transform = transforms.Compose([

    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224),

    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(45),

    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.3
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1,0.1),
        scale=(0.9,1.1)
    ),

    transforms.GaussianBlur(kernel_size=3),

    transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),
    
    transforms.RandomErasing(p=0.2)
])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    # ======================================
    # Dataset
    # ======================================

    dataset = datasets.ImageFolder(
        r"C:\Users\akash\NN lab\mini_project\dataset_expanded"
    )

    print("Total images:", len(dataset))
    print("Classes:", dataset.classes)

    num_classes = len(dataset.classes)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ======================================
    # Load SimCLR Encoder
    # ======================================

    encoder = Encoder().to(device)

    encoder.load_state_dict(
        torch.load("simclr_encoder.pth", weights_only=True)
    )

    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False

    # Fine-tune deeper layers of encoder
    for param in encoder.features[-4:].parameters():
        param.requires_grad = True

    # ======================================
    # Classifier
    # ======================================

    classifier = nn.Sequential(
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128,num_classes)
    ).to(device)

    model = nn.Sequential(encoder,classifier)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    scaler = torch.amp.GradScaler("cuda")  # updated AMP API

    # ======================================
    # Training
    # ======================================

    epochs = 35

    train_losses = []
    val_acc = []

    best_acc = 0

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images,labels in train_loader:

            images = images.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):

                outputs = model(images)
                loss = criterion(outputs,labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        train_losses.append(total_loss/len(train_loader))

        # ======================================
        # Validation
        # ======================================

        model.eval()

        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for images,labels in val_loader:

                images = images.to(device,non_blocking=True)
                labels = labels.to(device,non_blocking=True)

                outputs = model(images)

                _,preds = torch.max(outputs,1)

                correct += (preds==labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = 100*correct/total
        val_acc.append(acc)

        print(f"Epoch {epoch+1}/{epochs}  Loss:{train_losses[-1]:.3f}  Val Acc:{acc:.2f}%")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),"leaf_classifier.pth")
            print("Saved best model")

    print("Training complete")


    # ======================================
    # Classification Metrics
    # ======================================

    print("\nClassification Report:")
    print(classification_report(all_labels,all_preds,target_names=dataset.classes))


    # ======================================
    # Confusion Matrix
    # ======================================

    cm = confusion_matrix(all_labels,all_preds)

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        xticklabels=dataset.classes,
        yticklabels=dataset.classes
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")
    plt.show()


    # ======================================
    # Accuracy Plot
    # ======================================

    plt.plot(val_acc)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.savefig("accuracy_plot.png")
    plt.show()


    # ======================================
    # Loss Plot
    # ======================================

    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()


if __name__ == "__main__":
    main()
