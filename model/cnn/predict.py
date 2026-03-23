import torch
from torchvision import transforms
from PIL import Image
from .model import SimpleCNN

classes = ["bird", "drone"]

def predict_image(path):
    model = SimpleCNN()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)

    out = model(img)
    _, pred = torch.max(out, 1)

    return classes[pred.item()]   # ✅ FIX HERE