import torch
from torchvision import datasets, transforms
from model import CNNModel

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("data/spectrograms/train", transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = torch.utils.data.random_split(dataset,[train_size,val_size])

train_loader = torch.utils.data.DataLoader(train_ds,batch_size=8,shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds,batch_size=8)

model = CNNModel().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
loss_fn = torch.nn.CrossEntropyLoss()

best_acc = 0

for epoch in range(10):
    model.train()

    for x,y in train_loader:
        x,y = x.to(device), y.to(device)

        loss = loss_fn(model(x),y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    correct,total = 0,0
    model.eval()

    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            _,pred = torch.max(model(x),1)
            total += y.size(0)
            correct += (pred==y).sum().item()

    acc = 100*correct/total
    print(f"Epoch {epoch+1}: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(),"model/cnn/model.pth")

print("🔥 Best Accuracy:", best_acc)