# Flask Template

This sample repo contains the recommended structure for a Python Flask project. In this sample, we use `flask` to build a web application and the `pytest` to run tests.

 For a more in-depth tutorial, see our [Flask tutorial](https://code.visualstudio.com/docs/python/tutorial-flask).

 The code in this repo aims to follow Python style guidelines as outlined in [PEP 8](https://peps.python.org/pep-0008/).

## Running the Sample

To successfully run this example, we recommend the following VS Code extensions:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) 

- Open the template folder in VS Code (**File** > **Open Folder...**)
- Create a Python virtual environment using the **Python: Create Environment** command found in the Command Palette (**View > Command Palette**). Ensure you install dependencies found in the `pyproject.toml` file
- Ensure your newly created environment is selected using the **Python: Select Interpreter** command found in the Command Palette
- Run the app using the Run and Debug view or by pressing `F5`
- To test your app, ensure you have the dependencies from `dev-requirements.txt` installed in your environment
- Navigate to the Test Panel to configure your Python test or by triggering the **Python: Configure Tests** command from the Command Palette
- Run tests in the Test Panel or by clicking the Play Button next to the individual tests in the `test_app.py` file

# RAG Service

Este es un servicio Python independiente que proporciona funcionalidades de RAG (Retrieval-Augmented Generation) a través de endpoints HTTP. Está diseñado para ser consumido por el chatbot JavaScript.

## Características

- **API REST completa**: Endpoints HTTP para integración con aplicaciones frontend
- **Gestión de sesiones**: Mantiene el historial de conversaciones por sesión
- **Múltiples tipos de retriever**: TF-IDF, Dense (embeddings), Hybrid, y Re-ranking
- **Base de datos vectorial en memoria**: Usando FAISS para búsqueda eficiente
- **CORS habilitado**: Permite peticiones desde aplicaciones JavaScript
- **Generación con IA**: Integración con OpenAI y Together.ai

## Endpoints Disponibles

### POST `/api/rag/query`
Endpoint principal para consultas RAG desde el chatbot.

**Request:**
```json
{
  "query": "¿Cuál es la misión de HistoriaCard?",
  "top_k": 3,
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "success": true,
  "query": "¿Cuál es la misión de HistoriaCard?",
  "response": "La respuesta generada por IA...",
  "session_id": "uuid-session-id",
  "retrieved_documents": [...],
  "metadata": {
    "top_k": 3,
    "retriever_type": "tfidf",
    "num_chunks_indexed": 45
  }
}
```

### POST `/api/conversation/new`
Crear una nueva sesión de conversación.

### POST `/api/conversation/history`
Obtener el historial de una sesión específica.

### POST `/api/conversation/clear`
Limpiar el historial de una sesión.

### GET `/health`
Endpoint de health check para monitoreo.

### GET `/api/info`
Información del servicio y configuración.

## Instalación y Configuración

1. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

2. **Configurar variables de entorno:**
Crear un archivo `.env` basado en `.env.example`:
```bash
cp .env.example .env
```

3. **Ejecutar el servicio:**
```bash
python app.py
```

El servicio estará disponible en `http://localhost:5000`

## Variables de Entorno

- `RETRIEVER_TYPE`: Tipo de retriever (tfidf, dense, hybrid, rerank)
- `EMBEDDING_MODEL`: Modelo de embeddings para retrievers densos
- `USE_RERANKING`: Habilitar re-ranking (true/false)
- `TOGETHER_API_KEY`: API key para Together.ai
- `OPENAI_API_KEY`: API key para OpenAI (opcional)
- `PORT`: Puerto del servicio (default: 5000)
- `FLASK_DEBUG`: Modo debug (true/false)

## Arquitectura

```
rag-service/
├── app.py                 # Aplicación Flask principal
├── requirements.txt       # Dependencias Python
├── .env.example          # Ejemplo de configuración
├── rag/                  # Módulo RAG
│   ├── document_processor.py
│   ├── retriever.py
│   ├── generator.py
│   └── dialog_state.py
├── files/                # Documentos para indexar
└── context/             # Archivos de contexto
```

## Integración con Chatbot JavaScript

El chatbot JavaScript puede hacer peticiones HTTP al servicio:

```javascript
const response = await fetch('http://localhost:5000/api/rag/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: userMessage,
    session_id: currentSessionId
  })
});

const data = await response.json();
if (data.success) {
  const aiResponse = data.response;
  const sessionId = data.session_id;
  // Usar la respuesta en el chatbot
}
```

## Desarrollo

Para desarrollo local:
```bash
export FLASK_DEBUG=true
python app.py
```

El servicio se reiniciará automáticamente cuando detecte cambios en el código.
