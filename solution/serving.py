from torchvision import datasets, transforms

import mlflow
import PIL
import torch


device = torch.device("mps") # Change this with your device ("cpu", "cuda", "mps")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the image
image = PIL.Image.open("data/test_image.jpg")

# Preprocess the image
preprocessing=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
preprocessed_image = preprocessing(image)

loaded_model = mlflow.pytorch.load_model("models:/MyModel@champion")

predicted_probabilities = loaded_model(preprocessed_image.unsqueeze(0).to(device))
predicted_label = predicted_probabilities.argmax().cpu().numpy()

print(predicted_label)
