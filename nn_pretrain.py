# ==========================================
# SimCLR Self-Supervised Pretraining
# ==========================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

torch.backends.cudnn.benchmark = True


# ==========================================
# 1. SimCLR Augmentations
# ==========================================

simclr_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.4,0.4,0.4,0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()
])


# ==========================================
# 2. Dataset Class (must be top level)
# ==========================================

class SimCLRDataset(datasets.ImageFolder):

    def __getitem__(self, index):

        path, _ = self.samples[index]
        image = self.loader(path)

        xi = simclr_transform(image)
        xj = simclr_transform(image)

        return xi, xj


# ==========================================
# 3. Encoder Network
# ==========================================

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


# ==========================================
# 4. Projection Head
# ==========================================

class ProjectionHead(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64)
        )

    def forward(self,x):
        return self.net(x)


# ==========================================
# 5. Contrastive Loss
# ==========================================

def contrastive_loss(z_i,z_j,temperature=0.5):

    z_i = nn.functional.normalize(z_i,dim=1)
    z_j = nn.functional.normalize(z_j,dim=1)

    representations = torch.cat([z_i,z_j],dim=0)

    similarity_matrix = torch.matmul(
        representations,representations.T
    )

    batch_size = z_i.shape[0]

    positives = torch.cat([
        torch.diag(similarity_matrix,batch_size),
        torch.diag(similarity_matrix,-batch_size)
    ])

    mask = ~torch.eye(
        2*batch_size,
        dtype=bool,
        device=similarity_matrix.device
    )

    negatives = similarity_matrix[mask].view(
        2*batch_size,-1
    )

    logits = torch.cat([
        positives.unsqueeze(1),
        negatives
    ],dim=1)

    labels = torch.zeros(2*batch_size).long().to(similarity_matrix.device)

    logits = logits / temperature

    return nn.CrossEntropyLoss()(logits,labels)


# ==========================================
# 6. Training Function
# ==========================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    dataset = SimCLRDataset(
        root=r"C:\Users\akash\NN lab\mini_project\dataset_expanded"
    )

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    print("Total images:", len(dataset))
    print("Batches per epoch:", len(train_loader))

    encoder = Encoder().to(device)
    projector = ProjectionHead().to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(projector.parameters()),
        lr=0.001
    )

    epochs = 30

    for epoch in range(epochs):

        start_time = time.time()
        total_loss = 0

        for xi,xj in train_loader:

            xi = xi.to(device, non_blocking=True)
            xj = xj.to(device, non_blocking=True)

            hi = encoder(xi)
            hj = encoder(xj)

            zi = projector(hi)
            zj = projector(hj)

            loss = contrastive_loss(zi,zj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

    print("\n✅ SimCLR Pretraining Completed")

    torch.save(encoder.state_dict(),"simclr_encoder.pth")

    print("Encoder saved as simclr_encoder.pth")


# ==========================================
# Windows multiprocessing entry
# ==========================================

if __name__ == "__main__":
    main()