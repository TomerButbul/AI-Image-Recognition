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
