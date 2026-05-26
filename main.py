import torch
from PIL import Image
from torchvision import transforms
import numpy as np

from models.fastflow_model import FastFlow  # tu modelo
from datasets.preproccess import Preprocessing


# CONFIG

MODEL_PATH = "fastflow.pth"   # ruta a tu modelo entrenado
IMAGE_PATH = "datasets/Test_bearing_images/NG/2026_04_13_09_19_49_931-ng.jpg"  # imagen defectuosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# PREPROCESADO (igual que entrenamiento)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_image(path):
    image = Image.open(path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # (1, C, H, W)



# CARGAR MODELO

model = FastFlow().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()



# INFERENCIA

image = load_image(IMAGE_PATH).to(device)

with torch.no_grad():
    output = model(image)

    # log-prob total (sumando ramas)
    log_probs = []
    for log_pz, logdet in zip(output["log_prob_list"], output["logdet_list"]):
        log_probs.append(log_pz + logdet)

    total_log_prob = torch.stack(log_probs).sum()

    # Negative Log Likelihood (energía)
    nll = -total_log_prob.item()


failure_prob = 1 / (1 + np.exp(-nll / 1000))  # escala ajustable

print("Log-probabilidad total:", total_log_prob.item())
print("NLL (anomalía):", nll)
print("Probabilidad de fallo (aprox):", failure_prob)