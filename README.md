# Visual Search App

Sistema de búsqueda multimodal de imágenes que permite recuperar resultados usando texto o una imagen de referencia.

El proyecto implementa una arquitectura **multi-dataset**, usa embeddings de imágenes con **CLIP ViT-B/32**, consultas textuales multilingües con **SentenceTransformers**, búsqueda vectorial con **Qdrant**, backend con **FastAPI** y frontend con **Streamlit**.

---

## Resumen del proyecto

La aplicación permite realizar dos tipos de búsqueda:

```text
Texto → imagen
Imagen → imagen
```

La versión final trabaja con un índice multi-dataset de aproximadamente **95,607 imágenes**.

| Dataset                |               Dominio | Cantidad usada |
| ---------------------- | --------------------: | -------------: |
| Caltech-256            |     Objetos generales |         30,607 |
| Food-101               |                Comida |         20,000 |
| Fashion Product Images | Productos / ecommerce |         25,000 |
| Tiny ImageNet          |  Escalabilidad visual |         20,000 |

Total indexado:

```text
95,607 imágenes
```

---

## Arquitectura general

```text
Usuario
  |
  | Texto en español/inglés o imagen subida
  v
Streamlit Frontend
  |
  v
FastAPI Backend
  |
  v
EmbeddingService
  |-- Imagen: clip-ViT-B-32
  |-- Texto: sentence-transformers/clip-ViT-B-32-multilingual-v1
  |
  v
Qdrant
  |
  v
Resultados visuales multi-dataset
```

---

## Tecnologías utilizadas

* Python
* FastAPI
* Streamlit
* Qdrant
* SentenceTransformers
* PyTorch
* Hugging Face Datasets
* CLIP ViT-B/32
* NLTK / WordNet
* Docker

---

## Modelos utilizados

Modelo visual:

```text
clip-ViT-B-32
```

Modelo textual multilingüe:

```text
sentence-transformers/clip-ViT-B-32-multilingual-v1
```

Es el modelo que convierte consultas textuales en español, inglés u otros idiomas en embeddings compatibles con los embeddings visuales de las imágenes.

Ejemplos de consultas válidas:

```text
pizza
hamburguesa
zapatos
camiseta
perro
avión
mariposa
dog
camera
a photo of a motorcycle
```

---

## Configuración inicial

Crear el entorno virtual:

```powershell
python -m venv ..\venv
..\venv\Scripts\activate
```

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Descargar WordNet para mostrar nombres legibles en Tiny ImageNet:

```powershell
python -c "import nltk; nltk.download('wordnet')"
```

Copiar el archivo de variables de entorno:

```powershell
Copy-Item .env.example .env
```

---

## Levantar Qdrant

Ejecutar:

```powershell
docker compose up -d
```

Verificar que Qdrant responde:

```powershell
Invoke-RestMethod http://localhost:6333/collections
```

Para ver los nombres de las colecciones:

```powershell
(Invoke-RestMethod http://localhost:6333/collections).result.collections.name
```

---

## Flujo completo de ejecución de scripts

---

### 1. Exportar imágenes, miniaturas y manifiesto multi-dataset

```powershell
python -m scripts.export_multidataset
```

Este script descarga o reutiliza los datasets, exporta las imágenes originales, genera miniaturas y construye un manifiesto unificado.

Salida principal:

```text
data/manifests/visual_search_multidataset_manifest.csv
```

Resultado esperado:

```text
Procesadas en esta ejecución: 95,607
Registros totales en manifiesto: 95,607
EXPORTACIÓN MULTI-DATASET COMPLETADA
```

---

### 2. Generar embeddings de imágenes

```powershell
python -m scripts.generate_embeddings
```

Este script genera embeddings visuales para todas las imágenes del manifiesto multi-dataset usando `clip-ViT-B-32`.

Salida principal:

```text
data/embeddings/visual_search_multidataset_clip_vit_b32.npy
```

Resultado esperado:

```text
Forma: (95,607, 512)
Tipo: float32
Normas: 1.000000 – 1.000000
EMBEDDINGS MULTILINGÜES GENERADOS Y VALIDADOS
```

---

### 3. Verificar Qdrant

```powershell
python -m scripts.check_qdrant
```

Este script crea una colección temporal, inserta vectores de prueba y verifica que la búsqueda por similitud funcione correctamente.

---

### 4. Indexar embeddings en Qdrant

```powershell
python -m scripts.index_qdrant --recreate
```

Este script carga el archivo `.npy` de embeddings, lee el manifiesto multi-dataset e inserta los vectores en Qdrant.

Colección usada:

```text
visual_search_multidataset_v1
```

Resultado esperado:

```text
Puntos enviados: 95,607
Puntos en colección: 95,607
INDEXACIÓN MULTILINGÜE EN QDRANT COMPLETADA
```

---

### 5. Optimizar HNSW

```powershell
python -m scripts.enable_hnsw
```

Este script ayuda a que Qdrant termine la optimización del índice HNSW.

El estado `yellow` no necesariamente impide las búsquedas, pero para evaluación es recomendable intentar llegar a `green`.

---

## Ejecución de la aplicación

Para ejecutar el sistema completo se necesitan tres procesos: Qdrant, FastAPI y Streamlit.

---

### Terminal 1: Qdrant

```powershell
docker compose up -d
```

---

### Terminal 2: FastAPI

```powershell
python -m uvicorn src.api.main:app --reload
```

Documentación de la API:

```text
http://127.0.0.1:8000/docs
```

Endpoint de estado:

```text
http://127.0.0.1:8000/health
```

Endpoint de disponibilidad:

```text
http://127.0.0.1:8000/ready
```

---

### Terminal 3: Streamlit

```powershell
python -m streamlit run app.py
```

Interfaz web:

```text
http://localhost:8501
```

---

## Archivos generados localmente

La carpeta `data/` contiene archivos generados localmente.

```text
data/images/
data/thumbnails/
data/embeddings/
data/manifests/
data/cache/
```

---