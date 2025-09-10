AI Image Checker

This project contains a trained model for detecting AI-generated images.
⚠️ Note: The model was trained on images from earlier AI systems (e.g. ChatGPT-3 era). Newer image generators may bypass this checker due to changes in image quality.

🚀 How to Use

Install dependencies

pip install -r requirements.txt


Download the model weights

python download_model.py


Run the checker

python checker.py


This will launch a Gradio UI where you can upload images and see predictions (AI-generated vs Real).

🔧 Train Your Own Model

Want to improve accuracy against newer image generators?

Collect and label your own dataset (mix of real + AI images).

Place your dataset in the appropriate folder.

Run the trainer:

python trainer.py


This will create a new .pth file containing your trained weights.

Update checker.py to load your new model.

📂 Contents

checker.py → Gradio interface for checking images.

trainer.py → Script for training a new model.

download_model.py → Helper script to fetch pretrained weights.

requirements.txt → Python dependencies.
