import gradio as gr
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# -------------------------------------------------
# 1. Load model (cached)
# -------------------------------------------------
@torch.inference_mode()
def load_model():
    # Change filename if you use .pt instead of .pth
    model = torch.load("model.pth", map_location="cpu")
    model.eval()
    return model

model = load_model()

# -------------------------------------------------
# 2. Pre-processing (adjust to your training)
# -------------------------------------------------
INPUT_SIZE = 224  # change to 256, 384, etc. if needed

preprocess = transforms.Compose([
    transforms.Resize(int(INPUT_SIZE * 1.14)),   # e.g., 256 for 224
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats (common)
        std=[0.229, 0.224, 0.225],
    ),
])

# -------------------------------------------------
# 3. Class names – **EDIT THIS TO MATCH YOUR MODEL**
# -------------------------------------------------
# Example: ISIC 2019 (7 classes). Replace with your exact label order!
CLASS_NAMES = [
    "Melanocytic nevus",
    "Melanoma",
    "Benign keratosis",
    "Basal cell carcinoma",
    "Actinic keratosis",
    "Vascular lesion",
    "Dermatofibroma",
]

# -------------------------------------------------
# 4. Prediction function
# -------------------------------------------------
def predict(image: Image.Image):
    # Preprocess
    x = preprocess(image).unsqueeze(0)  # (1, C, H, W)

    # Inference
    logits = model(x)
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0)

    # Return dict for Gradio Label
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

# -------------------------------------------------
# 5. Gradio Interface
# -------------------------------------------------
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Upload Skin Lesion Image",
        tool="editor"
    ),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="DermAI – Skin Lesion Classifier",
    description=(
        "Upload a clear, well-lit photo of a skin lesion. "
        "Model runs on CPU. Results are **not medical advice**."
    ),
    examples=[
        ["examples/sample1.jpg"],
        ["examples/sample2.jpg"],
    ] if gr.examples else None,
    cache_examples=False,
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

# -------------------------------------------------
# 6. Launch
# -------------------------------------------------
if __name__ == "__main__":
    iface.launch()
