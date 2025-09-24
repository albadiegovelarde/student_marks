# Análisis del Rendimiento Estudiantil y Chatbot

Este proyecto proporciona un sistema para indexar, recuperar y analizar informes de rendimiento estudiantil utilizando búsqueda semántica y modelos de lenguaje (LLMs). Incluye generación automática de calificaciones, explicaciones y un chatbot interactivo para resúmenes por competencias.

---

## Tabla de Contenidos

- [Descripción](#descripción)
- [Requisitos](#requisitos)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Uso](#uso)
  - [1. Construir Índice](#1-construir-índice)
  - [2. Recuperar Información del Estudiante](#2-recuperar-información-del-estudiante)
  - [3. Generar Calificaciones y Explicaciones](#3-generar-calificaciones-y-explicaciones)
  - [4. Interfaz de Chatbot](#4-interfaz-de-chatbot)
- [Configuración] (#configuración)


---

## Descripción

El sistema consta de cuatro componentes principales:

1. **Indexación** (`indexer.py`): Lee informes estudiantiles en formato YAML, los divide en chunks, genera embeddings usando `sentence-transformers` y construye un índice FAISS para búsqueda semántica eficiente.

2. **Recuperación** (`retriever.py`): Carga el índice FAISS y permite recuperar los fragmentos más relevantes para un estudiante y consulta específica.

3. **Generación de Calificaciones** (`obtain_grade.py`): Usa los fragmentos recuperados y un modelo de lenguaje (`google/flan-t5-large`) para generar calificaciones automáticas, explicaciones y calcular la calificación global ponderada por estudiante.

4. **Chatbot** (`chatbot.py`): Proporciona una interfaz interactiva usando Gradio para consultar y resumir el desempeño del estudiante en competencias específicas.

---

## Requisitos

- Python 3.10+
- PyTorch
- Transformers
- SentenceTransformers
- FAISS
- Gradio
- PyYAML
- NumPy
- Pandas

Instalación de dependencias:

`pip install -r requirements.txt`

---

## Estructura del proyecto


project/
├── student_marks/
│ ├── indexer.py
│ ├── retriever.py
│ ├── obtain_grade.py
│ ├── chatbot.py
│ ├── skills.yaml
│ └── student_reports/
│ ├── student_1.yaml
│ └── student_2.yaml
├── notebooks/
│ ├── metrics.ipynb
│ ├── kmeans.ipynb
│ ├── orchestrator.ipynb
│ └── student_grades.json
└── README.md

---

## Uso

### 1. Construir Índice
`python student_marks/indexer.py`

- Lee todos los YAML en student_marks/student_reports/.
- Divide los informes en chunks con solapamiento.
- Genera embeddings usando sentence-transformers.
- Guarda el índice FAISS (student_marks/students_info.index) y metadata (student_marks/students_metadata.npy) en student_marks/.

### 2. Recuperar Información del Estudiante
`from student_marks.retriever import StudentRetriever

retriever = StudentRetriever()
resultados = retriever.retrieve(
    query="Math skills",
    student_id="student_1",
    top_k=3
)
print(resultados)
`

### 3. Generar Calificaciones y Explicaciones
`python student_marks/obtain_grade.py`

- Genera calificaciones y explicaciones usando los chunks recuperados.
- Calcula calificación global ponderada por estudiante.
- Devuelve un DataFrame de Pandas con calificaciones, explicaciones y fragmentos relevantes.

### 4. Interfaz de Chatbot
`python student_marks/chatbot.py`

- Lanza una interfaz Gradio.
- Ejemplo de pregunta: "Quiero un resumen del alumno student_1 de la skill matematicas"
- Devuelve un resumen conciso del desempeño del estudiante para esa competencia.

### 5. Notebooks Analíticos

- `notebooks/metrics.ipynb`: Análisis y métricas de las notas.
- `notebooks/kmeans.ipynb`: Clustering de estudiantes según sus calificaciones.
- `notebooks/orchestrator.ipynb`: Flujo de orquestación para generación de calificaciones y exportación.
- `notebooks/student_grades.json`: Archivo con calificaciones generadas por `obtain_grade.py`.


## Configuración

- Directorio de Informes: `student_marks/student_reports/`

- Archivo de Índice y Metadata: `students_info.index` y `students_metadata.npy` en `student_marks/`

- Modelo de Embeddings: `all-MiniLM-L6-v2` (configurable)

- Modelo LLM: `google/flan-t5-large` (configurable)

- Fragmentación de texto: `chunk_size` y `chunk_overlap` en `StudentIndexBuilder`.
