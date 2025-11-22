import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ltn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
data_root = "data/train/"  
batch_size = 32
n_epochs = 20
lr = 1e-4
img_size = 64  

class CNNModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),         
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),         
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          
        )
        flattened_dim = 64 * (img_size // 8) * (img_size // 8)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

cnn_model = CNNModel(in_channels=3).to(device)

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=data_root, transform=transform)
print("Classes encontradas:", train_dataset.classes)
assert train_dataset.classes[0] == "cats", "Classe 0 não é 'cats'"
assert train_dataset.classes[1] == "dogs", "Classe 1 não é 'dogs'"
cat_indices = [i for i, (_, y) in enumerate(train_dataset) if y == 0]
dog_indices = [i for i, (_, y) in enumerate(train_dataset) if y == 1]

cat_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=torch.utils.data.SubsetRandomSampler(cat_indices)
)

dog_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=torch.utils.data.SubsetRandomSampler(dog_indices)
)

Dog = ltn.Predicate(cnn_model)  

Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)

for epoch in range(n_epochs):
    cnn_model.train()
    train_loss = 0.0
    n_batches = 0

    for (dog_imgs, _), (cat_imgs, _) in zip(dog_loader, cat_loader):
        dog_imgs = dog_imgs.to(device)
        cat_imgs = cat_imgs.to(device)
        optimizer.zero_grad()
        dogs = ltn.Variable("dog", dog_imgs)
        cats = ltn.Variable("cat", cat_imgs)
        sat_agg = SatAgg(
            Forall(dogs, Dog(dogs)),
            Forall(cats, Not(Dog(cats)))
        )
        loss = 1.0 - sat_agg

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        n_batches += 1

    train_loss = train_loss / max(n_batches, 1)
    print(f"Epoch {epoch+1}/{n_epochs} - train_loss: {train_loss:.4f}")

torch.save(cnn_model.state_dict(), "cnn_dogs_cats_ltn.pth")
print("Treino finalizado e modelo salvo em cnn_dogs_cats_ltn.pth")
