import os

import torch
from PIL import Image
import torchvision.transforms as T

import gradio as gr
from gradio import themes

# Current path
current_path = os.getcwd()

# Define the model path
model_path = os.path.join(current_path, "ml", "models", "model.pt")

# Determine the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = torch.jit.load(model_path)
model.eval()  # Set to evaluation mode

# Depending on the device, load the model
model = model.to(device)


# Define the transformation
transform = T.Compose(
    [
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),  # Converts to [C, H, W] with values in [0, 1]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet mean  # ImageNet std
    ]
)


def cls_helper(label):
    if label == 0:
        return "Clear sky"
    elif label == 1:
        return "Cloudy"
    elif label == 2:
        return "Haze"
    else:
        return "Unknown"


def predict(image: Image.Image):
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        output = model(tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        pred_class = cls_helper(pred_idx)

    return pred_class


examples = [
    os.path.join(current_path, "ml", "data", "google_maps.jpg"),
    os.path.join(current_path, "ml", "data", "train_11890.jpg"),
    os.path.join(current_path, "ml", "data", "train_11716.jpg"),
]

theme = gr.Theme(
    primary_hue="blue",
    secondary_hue="blue",
    font="Arial",
    font_mono="Courier New",
)

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", height=350),
    outputs=["text"],
    examples=examples,
    title="Weather Condition Classifier",
    description="Upload an image to classify the weather condition as Clear sky, Cloudy, or Haze.",
    preload_example=0,
    theme=themes.Base(),
)

interface.launch(debug=True)
