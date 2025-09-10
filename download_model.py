import gdown  # pip install gdown

url = "https://drive.google.com/file/d/1ZUI6zJyClH_h2-1r7HF0aRpB89ytMWL3/view?usp=sharing"
output = "best_model.pth"
gdown.download(url, output, quiet=False)
