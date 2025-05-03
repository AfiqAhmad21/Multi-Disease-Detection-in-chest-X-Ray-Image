import torch
from torchvision import transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.models import mobilenet_v2
import torchvision.transforms.functional as F
from PIL import Image

class_names = ['COVID-19', 'Fibrosis', 'Normal', 'Pneumonia', 'Tuberculosis']

def load_model(weights_path="Model/mobilenet_lung_disease.pth"):
    model = mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_with_gradcam(model, img_pil):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    input_tensor = transform(img_pil).unsqueeze(0)
    input_tensor.requires_grad_()  # Enable gradient tracking for Grad-CAM

    # Hook Grad-CAM
    cam_extractor = GradCAM(model, target_layer="features.18.0")

    # Forward pass (DO NOT use torch.no_grad())
    output = model(input_tensor)

    # Get class index
    class_idx = output.squeeze().argmax().item()

    # Grad-CAM (backward + extract)
    activation_map = cam_extractor(class_idx, output)

    # Get heatmap overlay
    heatmap = overlay_mask(img_pil, F.to_pil_image(activation_map[0], mode='F'), alpha=0.5)

    # Softmax confidence
    confidence = torch.nn.functional.softmax(output, dim=1)[0][class_idx].item()
    label = class_names[class_idx]

    return label, confidence, heatmap

