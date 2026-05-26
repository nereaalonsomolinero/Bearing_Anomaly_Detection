from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

class Preprocessing(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ) #Normalizamos siguiendo los mismos criterios que en el dataset ImageNet
        ])

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.image_paths = [
            p for p in self.root_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]

        print("Número de imágenes encontradas:", len(self.image_paths))
        

        if len(self.image_paths) == 0:
            raise ValueError(f"No se encontraron imágenes en {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path)
        
        image = Image.open(image_path).convert("RGB")

        image = image.crop((270, 225, 935, 900))
        
        image = self.transform(image)

        return image, str(image_path)


#hago la prueba de que el preproecsamiento hecho funciona correctamente sobre nuestro dataset. 
def imshow(img_tensor, title=""):
    img = img_tensor.numpy().transpose((1, 2, 0))

    # Desnormalizar
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.title(title)
    plt.axis("off")


if __name__ == "__main__":
    dataset = Preprocessing(
        root_dir="datasets/Train_bearing_images",
        image_size=256
    )

    # #Cargar una imagen directamentepara ver su estructura original
    # raw_path = dataset.image_paths[0]
    # raw_image = Image.open(raw_path)

    # print("IMAGEN ORIGINAL")
    # print("Tamaño:", raw_image.size)
    # print("Modo (canales):", raw_image.mode)

    # plt.figure(figsize=(5, 5))
    # plt.imshow(raw_image, cmap='gray')
    # plt.title("Imagen original")
    # plt.axis("off")

    # #Imagen preprocesada
    # processed_image, path = dataset[0]

    # print("IMAGEN PREPROCESADA")
    # print("Shape tensor:", processed_image.shape)  # (C, H, W)

    # plt.figure(figsize=(5, 5))
    # imshow(processed_image, "Imagen preprocesada")

    # # --- DataLoader ---
    # loader = DataLoader(dataset, batch_size=4, shuffle=True)
    # images, paths = next(iter(loader))

    # print("BATCH")
    # print("Shape del batch:", images.shape)  # (B, C, H, W)
    # print("Primera ruta:", paths[0])

    # plt.show()

    output_dir = Path("datasets/preprocess/OK")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGuardando todas las imágenes preprocesadas en:", output_dir)

    for i in range(len(dataset)):
        img_tensor, path = dataset[i]

        # Convertir tensor → imagen (desnormalizar primero)
        img = img_tensor.numpy().transpose((1, 2, 0))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        # Convertir a formato PIL
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)

        # Nombre de salida
        original_name = Path(path).name
        save_path = output_dir / original_name

        # Guardar
        img_pil.save(save_path)

        if i % 100 == 0:
            print(f"{i}/{len(dataset)} imágenes guardadas")

    print("✅ Todas las imágenes han sido preprocesadas y guardadas ")

#para ver donde se entrena añadir: # 5) DISPOSITIVO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Dispositivo de entrenamiento:", device)and


#esta creo qeu se puede suprimir, estaba en la clase de arriba  
def anomaly_map(self, x, out_size=(256, 256)):
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
            maps = []
            for feat, adapter, flow in zip(features, self.adapters, self.flow_branches):
                feat_adapted = adapter(feat)
                z, _ = flow(feat_adapted)
                energy = torch.mean(z ** 2, dim=1, keepdim=True)
                energy_up = F.interpolate(
                    energy, size=out_size, mode="bilinear", align_corners=False
                )
                maps.append(energy_up)
            return torch.mean(torch.stack(maps, dim=0), dim=0)
        



#codiog que muestra las images mas anomalas para el modelo y las mas normales vs su preccion real:sorted_idx = np.argsort(all_scores)
print("5 imágenes más normales según el modelo")
for idx in sorted_idx[:5]:
    print(f"Score={all_scores[idx]:.6f} | True={all_labels[idx]} | Pred={pred_labels[idx]} | Path={all_paths[idx]}")

print("5 imágenes más anómalas según el modelo")
for idx in sorted_idx[-5:][::-1]:
    print(f"Score={all_scores[idx]:.6f} | True={all_labels[idx]} | Pred={pred_labels[idx]} | Path={all_paths[idx]}")
