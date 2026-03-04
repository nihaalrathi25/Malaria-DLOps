import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.model.model import MalariaCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
class_names = ["Parasitized", "Uninfected"]

# Load model
model = MalariaCNN()
model.load_state_dict(
    torch.load("artifacts/models/malaria_model_v2.pth", map_location=device)
)
model.to(device)
model.eval()

# IMPORTANT: Must match training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return class_names[predicted.item()], confidence.item()