"""
Módulo de preprocesamiento para imágenes de rodamientos.

Detecta automáticamente la región de interés circular mediante
la transformada de Hough, recorta el rodamiento y aplica el preprocesamiento
estándar de ImageNet requerido por Wide ResNet-50.

"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Union



# Parámetros por defecto
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225] #usamos el preprocesamiento estandar de ImageNet

DEFAULT_OUTPUT_SIZE = 224          # resolución esperada por Wide ResNet-50
DEFAULT_MARGIN_RATIO = 0.05        # margen extra alrededor del círculo (5 %)
DEFAULT_HOUGH_PARAM1 = 100         # umbral alto del detector Canny interno
DEFAULT_HOUGH_PARAM2 = 30          # umbral de acumulador de Hough
DEFAULT_MIN_RADIUS_RATIO = 0.15    # radio mínimo relativo al lado corto
DEFAULT_MAX_RADIUS_RATIO = 0.48    # radio máximo relativo al lado corto



# Función detección del círculo del rodamiento
def detect_bearing_circle(
    gray: np.ndarray,
    param1: float = DEFAULT_HOUGH_PARAM1,
    param2: float = DEFAULT_HOUGH_PARAM2,
    min_radius_ratio: float = DEFAULT_MIN_RADIUS_RATIO,
    max_radius_ratio: float = DEFAULT_MAX_RADIUS_RATIO,
) -> Optional[Tuple[int, int, int]]:
    
    h, w = gray.shape[:2]
    
    # 🔥 OPTIMIZACIÓN CLAVE: Reducir tamaño para la detección
    MAX_DIM = 640 
    scale = 1.0
    
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        # Redimensionamos la imagen a un tamaño manejable
        gray_proc = cv2.resize(gray, (int(w * scale), int(h * scale)))
    else:
        gray_proc = gray

    # Recalculamos parámetros basados en la imagen reducida
    short_side = min(gray_proc.shape[:2])
    min_r = int(short_side * min_radius_ratio)
    max_r = int(short_side * max_radius_ratio)

    # Suavizado ligero para reducir ruido antes de Hough
    blurred = cv2.GaussianBlur(gray_proc, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=short_side // 4,
        param1=param1,
        param2=param2,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None

    # Seleccionar el círculo con mayor radio
    circles = np.round(circles[0, :]).astype(int)
    best = circles[np.argmax(circles[:, 2])]
    cx, cy, r = int(best[0]), int(best[1]), int(best[2])
    
    # 🔥 Revertimos la escala a las dimensiones originales
    if scale != 1.0:
        cx = int(cx / scale)
        cy = int(cy / scale)
        r  = int(r / scale)
        
    return cx, cy, r

# Función para recortar la imágen con forma cuadrada alrededor del círculo detectado
def crop_bearing_roi(
    image: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
) -> np.ndarray:
    """
    Extrae una región cuadrada alreddedor del circulo mas grande.
    """
    h, w = image.shape[:2]
    margin = int(r * margin_ratio)
    side   = r + margin

    x0 = max(0, cx - side)
    y0 = max(0, cy - side)
    x1 = min(w, cx + side)
    y1 = min(h, cy + side)

    return image[y0:y1, x0:x1]



# Clase principal de preprocesamiento
class BearingPreprocessor:

    def __init__(self, output_size=224):
        self.output_size = output_size

    # -------------------------------------------------
    # PROCESS SINGLE IMAGE
    # -------------------------------------------------
    def process_image(self, image_bgr):
        """Devuelve SOLO lo necesario por imagen (sin guardar)"""

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        result = detect_bearing_circle(gray)

        annotated = image_rgb.copy()

        if result:
            cx, cy, r = result
            cv2.circle(annotated, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 5, (255, 0, 0), -1)
            cropped = crop_bearing_roi(image_rgb, cx, cy, r)
        else:
            h, w = image_rgb.shape[:2]
            s = min(h, w)
            cropped = image_rgb[:s, :s]

        resized = np.array(Image.fromarray(cropped).resize((224, 224)))

        return image_rgb, annotated, cropped, resized


    # -------------------------------------------------
    # STREAMING PIPELINE (NO RAM USAGE)
    # -------------------------------------------------
    def process_folder(self, folder_path: Union[str, Path]):
        folder = Path(folder_path)
        images = list(folder.glob("*.*"))

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            original, annotated, cropped, resized = self.process_image(img)

            # 🔥 NO guardamos nada → lo emitimos y se destruye
            yield {
                "name": img_path.name,
                "original": original,
                "hough": annotated,
                "cropped": cropped,
                "final": resized
            }





if __name__ == "__main__":
    print("Iniciando prueba...")
    train_images = "datasets/Train_bearing_images"

    pre = BearingPreprocessor(output_size=224)
    print("⚙️ preprocesador instanciado")
    
    # Nota: Eliminada la línea "results = pre.process_folder" suelta que no hacía nada
    
    for i, item in enumerate(pre.process_folder(train_images)):
        if i >= 5:
            break

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(item["name"])

        axs[0].imshow(item["original"])
        axs[0].set_title("Original")
        axs[1].imshow(item["hough"])
        axs[1].set_title("Detección Hough")
        axs[2].imshow(item["cropped"])
        axs[2].set_title("Recorte")
        axs[3].imshow(item["final"])
        axs[3].set_title("Final (224x224)")

        for ax in axs:
            ax.axis("off")

        # 🔥 Mostrar sin bloquear, pausar 2 segundos y cerrar
        plt.show(block=False)
        plt.pause(2.5) # Pausa para que te dé tiempo a verlo
        plt.close(fig) 
        
    print("✅ Prueba completada.")