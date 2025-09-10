import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Define model with SAME head as trainer.py ===
model = models.resnet50(weights=None)  # don't load pretrained weights
model.fc = nn.Sequential(
    nn.Linear(2048, 128),  # fc.0
    nn.ReLU(),             # fc.1
    nn.Linear(128, 1)      # fc.2
)


# === Load trained weights ===
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# === Image preprocessing (MUST match trainer transforms) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Prediction function ===
def predict_image(image: Image.Image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob_ai = torch.sigmoid(output).item()
    label = "AI-generated" if prob_ai >= 0.0003 else "Real"
    confidence = f"{(prob_ai if label=='AI-generated' else 1-prob_ai)*100:.2f}%"
    return label, confidence

# === Gradio UI ===
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(), gr.Textbox(label="Confidence")],
    title="AI Image Checker",
    description="Upload an image to check if it's AI-generated or Real."
)

interface.launch()
