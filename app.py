import os
from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
from get_plant_info import get_plant_info

app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the model
MODEL_PATH = r'C:\Users\Amulya H G\OneDrive\Desktop\PC1 Desktop data\BioBotanica\models\resnet50.pth'

# Class names based on the training dataset
class_names = ["Aloe Vera", "Amla", "Amruta Balli", "Arali", "Hibiscus", "Lemon Grass", "Mint"]

# Configure upload folder for storing the image temporarily
UPLOAD_FOLDER = r'C:\Users\Amulya H G\Desktop\PC1 Desktop data\BioBotanica\static\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
def load_model(num_classes):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    # Explicitly load only the weights
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)


    model.to(device)
    model.eval()
    return model

model = load_model(num_classes=len(class_names))

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        uploaded_file = request.files['image']
        if uploaded_file:
            # Save the uploaded file temporarily with a unique filename
            filename = uploaded_file.filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(image_path)
            
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Preprocess the image
            input_image = transform(image).unsqueeze(0).to(device)
            
            # Perform inference
            with torch.no_grad():
                output = model(input_image)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class_idx = predicted.item()
                predicted_class_name = class_names[predicted_class_idx]

            # Confidence threshold
            confidence_threshold = 0.6
            if confidence.item() < confidence_threshold:
                return render_template(
                    "index.html",
                    result="Plant Not Found",
                    confidence="Low Confidence",
                    uploaded_image=None
                )
            
            # Fetch plant info
            medicinal_info = get_plant_info(predicted_class_name)
            return render_template(
                "index.html",
                result=predicted_class_name,
                confidence=f"{confidence.item() * 100:.2f}%",
                info=medicinal_info,
                uploaded_image=f"uploads/{filename}"  # Relative path for the <img> tag
            )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

#to run this-python app.py
