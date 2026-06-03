"""# Detección de Anomalías Industriales mediante FastFlow y WideResNet

## 📌 Descripción del Proyecto
Este repositorio contiene la implementación de un pipeline avanzado de **Detección y Localización de Anomalías en Entornos Industriales** utilizando la arquitectura **FastFlow** apoyada por un extractor de características (Backbone) **WideResNet-50-2** preentrenado.

El objetivo principal es identificar y generar mapas de calor precisos sobre imperfecciones, fallos o elementos extraños en piezas metálicas y componentes complejos (como rodillos), optimizando tanto la tasa de verdaderos positivos como la mitigación de falsos positivos provocados por reflejos e iluminaciones complejas del material.

## 🏗️ Arquitectura del Pipeline

El flujo completo de procesamiento de datos y modelado se divide en cuatro etapas críticas:

### 1. Preprocesamiento y Enmascaramiento Dinámico
Para resolver la problemática de las malas clasificaciones provocadas por el **brillo intrínseco del metal**, se integra un módulo de enmascaramiento personalizado inmediatamente después del redimensionado de las imágenes:
* **Redimensionamiento:** Todas las imágenes de entrada se unifican a un tamaño estándar de $3 \\times 224 \\times 224$.
* **Normalización:** Escalado de píxeles mediante división por 255 y estandarización usando la media y desviación estándar correspondientes.
* **Máscaras de Región de Interés (RoI):** Se aplican máscaras binarias para excluir las zonas interiores y exteriores de los rodillos que tienden a generar falsas alarmas debido a destellos lumínicos. Con esto, el modelo concentra su atención exclusivamente en la superficie de inspección útil.

### 2. Extractor de Características Congelado (WideResNet Backbone)
Se utiliza una red **WideResNet-50-2** precargada con pesos de ImageNet. Durante toda la fase de entrenamiento de FastFlow, esta red permanece completamente **congelada** (`requires_grad=False`). Sus pesos no varían, actuando como un extractor de descriptores visuales fijos de alta densidad.
Se extraen de forma jerárquica tres niveles de mapas de características (Multiscale Feature Extraction):
* **`layer1`:** Características de bajo nivel (texturas finas, bordes microestructurales). Dimensión espacial y canales: $256 \\times 56 \\times 56$.
* **`layer2`:** Características de nivel medio (formas combinadas, patrones locales). Dimensión espacial y canales: $512 \\times 28 \\times 28$.
* **`layer3`:** Características de alto nivel (contexto global, relaciones semánticas abstractas). Dimensión espacial y canales: $1024 \\times 14 \\times 14$.
* *Nota:* Se descarta deliberadamente la capa `layer4` para evitar una pérdida excesiva de resolución espacial que perjudicaría la localización fina píxel a píxel.

### 3. Módulo FastFlow (Flujos de Normalización 2D)
Cada una de las tres escalas extraídas es proyectada de manera independiente a través de bloques invertibles bidimensionales:
* **Adaptación de Canales (Convolución 1x1):** Unifica las diferentes profundidades de canales (256, 512, 1024) a una dimensión oculta homogénea sin alterar el tamaño espacial de los mapas.
* **Pasos de Flujo (Flow Steps):** Compuesto por un encadenamiento de Bloques de Acoplamiento Afín 2D.
  * **División (Split):** Los canales se segmentan exactamente en dos mitades ($x_1, x_2$).
  * **Sub-red Convolucional:** La mitad $x_1$ pasa por convoluciones alternadas de $3\\times3$ y $1\\times1$ para predecir coeficientes de escala ($s$) y traslación ($t$).
  * **Transformación Afín:** La segunda mitad se transforma mediante la ecuación de acoplamiento: $y_2 = x_2 \\odot \\exp(s(x_1)) + t(x_1)$ mientras que la primera mitad fluye intacta ($y_1 = x_1$).
  * **Permutación de Canales:** Una convolución $1\\times1$ invertible mezcla los canales para asegurar una interconectividad completa de dimensiones en el siguiente paso de flujo.

### 4. Inferencia y Generación del Mapa de Anomalías
* **Mapeo en Espacio Latente ($z$):** El flujo proyecta los datos normales hacia una distribución Gaussiana estándar perfecta. Las anomalías, al no seguir esta distribución, caen en zonas de baja verosimilitud (alta energía anómala).
* **Interpolación Bilineal (Upsampling):** Los mapas de anomalías calculados a diferentes escalas reducidas se reescalan matemáticamente de vuelta a las dimensiones originales de la imagen ($224 \\times 224$).
* **Fusión Multiescala:** Se promedian los mapas de calor correspondientes para consolidar una única matriz bidimensional que localiza con exactitud milimétrica el defecto sobre la pieza real.