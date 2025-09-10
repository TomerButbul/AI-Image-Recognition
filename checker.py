import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import gradio as gr

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model
model = models.resnet50(weights=None)  # We are loading custom weights
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1)
)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction function with correct confidence
def predict_image(image: Image.Image):
    try:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)  # Add batch dim

        with torch.no_grad():
            output = model(tensor)
            prob_ai = torch.sigmoid(output).item()

        if prob_ai >= 0.0003:
            label = "AI-generated"
            confidence = f"{prob_ai*100:.2f}%"
        else:
            label = "Real"
            confidence = f"{(1 - prob_ai)*100:.2f}%"

        return label, confidence

    except Exception as e:
        return "Error", str(e)

# Gradio Interface
description = "Upload an image, and the AI will determine if it is AI-generated or real."
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[gr.Text(label="Prediction"), gr.Text(label="Confidence")],
    title="AI Image Detector",
    description=description
)

if __name__ == "__main__":
    interface.launch(share=True)
