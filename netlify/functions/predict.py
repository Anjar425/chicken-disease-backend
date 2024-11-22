# api/predict.py
import json
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO

# Load the model
model = torch.nn.Module()
model_selection = "densenet"
num_classes = 4
if model_selection == "densenet":
    import torchvision
    model = torchvision.models.densenet121(pretrained=True)  # Menggunakan DenseNet121
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()  # Pastikan model dalam mode evaluasi

# Preprocessing untuk gambar
def preprocess(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize ke ukuran input model
        transforms.ToTensor(),          # Ubah gambar jadi tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisasi
    ])
    return transform(image).unsqueeze(0)

def handler(request):
    try:
        # Ambil gambar dari request
        image_bytes = request.body
        img_tensor = preprocess(image_bytes)

        # Prediksi dengan model
        with torch.no_grad():
            output = model(img_tensor)

        # Ambil hasil prediksi
        _, predicted_class = torch.max(output, 1)

        # Return response
        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": predicted_class.item()}),
            "headers": {"Content-Type": "application/json"}
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {"Content-Type": "application/json"}
        }
