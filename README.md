# Visual Search App — Búsqueda multimodal de imágenes

Sistema de búsqueda multimodal que permite recuperar imágenes mediante texto o una imagen de referencia.
El proyecto usa el dataset **Caltech-256**, genera embeddings visuales con **CLIP ViT-B/32**, procesa consultas textuales con un modelo **CLIP multilingüe** y almacena los vectores en **Qdrant** para búsqueda por similitud.

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
Resultados visuales de Caltech-256
```

## Tecnologías principales

* Python
* FastAPI
* Streamlit
* Qdrant
* SentenceTransformers
* CLIP ViT-B/32
* PyTorch
* Hugging Face Datasets
* Caltech-256

## Dataset y modelo

El dataset utilizado es:

```text
ilee0022/Caltech-256
```

El modelo de texto:

```text
sentence-transformers/clip-ViT-B-32-multilingual-v1
```

Este modelo permite transformar consultas en español, inglés u otros idiomas en embeddings compatibles con los embeddings visuales de las imágenes.

Ejemplos de consultas válidas:

```text
avión
airplane
una foto de una motocicleta
a photo of a motorcycle
mariposa
butterfly
```

## Configuración del entorno

Crear y activar entorno virtual:

```powershell
python -m venv ..\venv
..\venv\Scripts\activate
```

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Copiar archivo de variables de entorno:

```powershell
Copy-Item .env.example .env
```

## Levantar Qdrant

Antes de indexar o buscar imágenes, Qdrant debe estar activo:

```powershell
docker compose up -d
```

Verificar que Qdrant responde:

```powershell
Invoke-RestMethod http://localhost:6333/collections
```

Para detener Qdrant:

```powershell
docker compose down
```

## Flujo completo de ejecución de scripts

### 1. Inspeccionar el dataset

```powershell
python -m scripts.inspect_dataset
```

Este script descarga o reutiliza el dataset desde Hugging Face, revisa su estructura, valida columnas esperadas y genera un reporte del dataset.

Salidas principales:

```text
data/manifests/dataset_report.json
data/manifests/samples/
```

### 2. Exportar imágenes, miniaturas y manifiesto

```powershell
python -m scripts.export_dataset
```

Este script exporta las imágenes originales, genera miniaturas y construye el manifiesto principal del dataset.

Salidas principales:

```text
data/images/
data/thumbnails/
data/manifests/caltech256_manifest.csv
data/manifests/caltech256_export_progress.jsonl
```

### 3. Verificar la exportación

```powershell
python -m scripts.verify_export
```

Este script valida que las imágenes, miniaturas, rutas, tamaños, identificadores y manifiestos estén correctos.

Verifica principalmente:

```text
data/images/
data/thumbnails/
data/manifests/caltech256_manifest.csv
```

### 4. Generar embeddings de imágenes

```powershell
python -m scripts.generate_embeddings
```

Este script carga las imágenes del manifiesto y genera embeddings visuales de dimensión 512 usando `clip-ViT-B-32`.

Salida principal:

```text
data/embeddings/caltech256_multilingual_clip_vit_b32.npy
```

También genera reportes de progreso:

```text
data/manifests/embedding_progress_multilingual.json
data/manifests/embedding_report_multilingual.json
```

### 5. Verificar conexión con Qdrant

```powershell
python -m scripts.check_qdrant
```

Este script crea una colección temporal en Qdrant, inserta vectores de prueba y verifica que la búsqueda por similitud funcione correctamente.

### 6. Indexar embeddings en Qdrant

```powershell
python -m scripts.index_qdrant --recreate
```

Este script carga el archivo `.npy` de embeddings, lee los metadatos del manifiesto y crea la colección vectorial en Qdrant.

Colección usada:

```text
caltech256_multilingual_v1
```

Salida principal:

```text
data/manifests/qdrant_index_report_multilingual.json
```

### 7. Activar u optimizar HNSW en Qdrant

```powershell
python -m scripts.enable_hnsw
```

Este script ajusta la configuración de indexación HNSW y espera a que Qdrant termine de optimizar la colección.

### 8. Probar búsqueda multimodal

```powershell
python -m scripts.check_multimodal_search
```

Este script prueba dos tipos de búsqueda:

```text
Texto → imagen
Imagen → imagen
```

También calcula métricas simples como `Precision@K` para validar que el sistema recupera imágenes coherentes.

Salida principal:

```text
data/manifests/multimodal_search_report.json
```


## Ejecución de la aplicación

Una vez generados los embeddings e indexados en Qdrant, se debe levantar el backend y el frontend.

### Terminal 1: Qdrant

```powershell
docker compose up -d
```

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

### Terminal 3: Streamlit

```powershell
python -m streamlit run app.py
```

Interfaz web:

```text
http://localhost:8501
```