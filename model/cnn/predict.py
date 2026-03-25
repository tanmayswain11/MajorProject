import torch
from torchvision import transforms
from PIL import Image
from .model import CNNModel

classes = ["bird","drone","plane"]

def predict_image(path):
    model = CNNModel()
    model.load_state_dict(torch.load("model/cnn/model.pth"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        _,pred = torch.max(model(img),1)

    return classes[pred.item()]