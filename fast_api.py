from flask import Flask, request, render_template, redirect, url_for
import uuid
import os
import torch
import numpy as np
from PIL import Image
from models.attention_unet import AttentionUNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import GreenChannelCLAHE  # Replace with your actual module path

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Attention U-Net model
model_sclera = AttentionUNet()
model_sclera.load_state_dict(torch.load(
    "/home/teaching/DL_Hack/trained-models/model_50_attention.pth", map_location=device))
model_sclera.to(device)
model_sclera.eval()

# Preprocessing transforms
transform = A.Compose([
    A.Resize(720, 720),
    GreenChannelCLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    ToTensorV2(),
])

# Sclera segmentation function
def segment_sclera(img_tensor):
    with torch.no_grad():
        output = model_sclera(img_tensor)
        sclera_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        sclera_mask = (sclera_mask > 0.5).astype(np.uint8)
    return sclera_mask

# Routes
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    if not file:
        return redirect(url_for("home"))

    # Save original image
    img = Image.open(file).convert("RGB")
    filename = f"{uuid.uuid4().hex}.png"
    original_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    img.save(original_path)

    # Preprocess image (corrected line: ensure float32)
    input_tensor = transform(image=np.array(img))["image"].unsqueeze(0).float().to(device)

    # Run model
    sclera_mask = segment_sclera(input_tensor)

    # Save mask
    sclera_mask_img = Image.fromarray((sclera_mask * 255).astype("uint8"))
    sclera_mask_path = os.path.join(app.config["UPLOAD_FOLDER"], f"sclera_{filename}")
    sclera_mask_img.save(sclera_mask_path)

    return render_template("results.html",
                           original_image=f"uploads/{filename}",
                           sclera_image=f"uploads/sclera_{filename}")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
