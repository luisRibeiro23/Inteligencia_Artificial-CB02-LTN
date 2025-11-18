import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ltn


# =========================
# 1. Configurações básicas
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data_root = "data/train"   # pasta com subpastas: data/train/cats e data/train/dogs
batch_size = 32
n_epochs = 20
lr = 1e-4
img_size = 64  # ajuste se seu dataset tiver outro tamanho


# =========================
# 2. Modelo CNN (predicado Dog)
# =========================
class CNNModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),           # H,W -> H/2, W/2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),           # H,W -> H/4, W/4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),           # H,W -> H/8, W/8
        )
        # se img_size = 64, depois de 3 pools (÷2, ÷2, ÷2) -> 8x8
        flattened_dim = 64 * (img_size // 8) * (img_size // 8)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()               # saída em [0,1] = grau de verdade Dog(x)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # shape: (batch, 1)


cnn_model = CNNModel(in_channels=3).to(device)


# =================================
# 3. DataLoader (dogs e cats)
# =================================
# Transform para normalizar as imagens
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),  # converte para [0,1] e (C,H,W)
])

# Espera:
# data_root/
#   cats/
#       *.jpg / *.png ...
#   dogs/
#       *.jpg / *.png ...
train_dataset = datasets.ImageFolder(root=data_root, transform=transform)

print("Classes encontradas:", train_dataset.classes)
# Idealmente: ['cats', 'dogs']
# cats -> label 0, dogs -> label 1

# Separa índices de cada classe
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

# Wrapper pra ter exatamente o formato do paper:
# train_dataloader -> batches de (dog_imgs, cat_imgs)
class DogsCatsPairLoader:
    def __init__(self, dog_loader, cat_loader):
        self.dog_loader = iter(dog_loader)
        self.cat_loader = iter(cat_loader)
        self.length = min(len(dog_loader), len(cat_loader))

    def __iter__(self):
        # recria iteradores a cada época
        self.dog_iter = iter(self.dog_loader)
        self.cat_iter = iter(self.cat_loader)
        for _ in range(self.length):
            try:
                dog_imgs, _ = next(self.dog_iter)
            except StopIteration:
                self.dog_iter = iter(self.dog_loader)
                dog_imgs, _ = next(self.dog_iter)

            try:
                cat_imgs, _ = next(self.cat_iter)
            except StopIteration:
                self.cat_iter = iter(self.cat_loader)
                cat_imgs, _ = next(self.cat_iter)

            yield dog_imgs, cat_imgs

    def __len__(self):
        return self.length


train_dataloader = DogsCatsPairLoader(dog_loader, cat_loader)


# ===========================================
# 4. LTNtorch – implementação da Seção 3.1
# ===========================================
# define the Dog predicate
Dog = ltn.Predicate(cnn_model)

# define logical operators and formula aggregator (SatAgg)
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()

# optimizer
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)

# training loop
for epoch in range(n_epochs):
    cnn_model.train()
    train_loss = 0.0
    n_batches = 0

    for i, (dog_imgs, cat_imgs) in enumerate(train_dataloader):
        dog_imgs = dog_imgs.to(device)
        cat_imgs = cat_imgs.to(device)

        optimizer.zero_grad()

        # ground logical variables with current training batch
        dogs = ltn.Variable("dog", dog_imgs)   # positive examples (dogs)
        cats = ltn.Variable("cat", cat_imgs)   # negative examples (cats)

        # compute loss function
        # phi1 = Forall(dog, Dog(dog))
        # phi2 = Forall(cat, Not(Dog(cat)))
        sat_agg = SatAgg(
            Forall(dogs, Dog(dogs)),        # this is φ1
            Forall(cats, Not(Dog(cats)))    # this is φ2
        )

        loss = 1.0 - sat_agg   # L(θ) = 1 − SatAgg

        # back-propagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        n_batches += 1

    train_loss = train_loss / max(n_batches, 1)
    print(f"Epoch {epoch+1}/{n_epochs} - train_loss: {train_loss:.4f}")

