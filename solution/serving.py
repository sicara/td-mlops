from fastapi import FastAPI, UploadFile
from torchvision import datasets, transforms

import mlflow
import PIL
import torch


app = FastAPI()

mlflow.set_tracking_uri("http://127.0.0.1:5000")

device = torch.device("mps") # Change this with your device ("cpu", "cuda", "mps")

@app.post("/predict")
async def predict(file: UploadFile):
    # Load the image
    image = PIL.Image.open(file.file)

    # Transform the image

    # Preprocess the image
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    preprocessed_image = preprocessing(image)

    loaded_model = mlflow.pytorch.load_model("models:/MyModel@champion")

    predicted_probabilities = loaded_model(preprocessed_image.unsqueeze(0).to(device))
    predicted_label = int(predicted_probabilities.argmax().cpu())

    return {"prediction": predicted_label}
