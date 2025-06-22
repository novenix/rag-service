from flask import Flask, request, jsonify
from flask_cors import CORS
from rag.document_processor import DocumentProcessor
from rag.retriever import get_retriever, TFIDFRetriever, DenseRetriever
from rag.generator import get_generator
from rag.dialog_state import DialogStateManager
import os
import requests
from apscheduler.schedulers.background import BackgroundScheduler
import threading
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for JavaScript frontend
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev_secret_key')

# Initialize RAG components
documents_dir = os.path.join(os.path.dirname(__file__), 'files')
processor = DocumentProcessor(documents_dir)

# Initialize the dialog state manager
dialog_manager = DialogStateManager()

# Get configuration from environment
retriever_type = os.getenv('RETRIEVER_TYPE', 'tfidf')
embedding_model = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
use_reranking = os.getenv('USE_RERANKING', 'false').lower() == 'true'

# Get retriever based on configuration
retriever_kwargs = {}
print(f"Using use_reranking type: {use_reranking}")
if retriever_type.lower() == 'dense':
    retriever_kwargs = {
        'model_name': embedding_model,
        'vector_store_config': {'index_type': 'flat'}
    }
    print(f"Using dense retriever with embedding model: {embedding_model}")
    if use_reranking:
        retriever_type = 'rerank'
        retriever_kwargs = {
            'base_retriever_type': 'dense',
            'base_retriever_kwargs': retriever_kwargs,
            'cross_encoder_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'initial_top_k': 10
        }
        
elif retriever_type.lower() == 'hybrid':
    # Create individual retrievers
    tfidf_retriever = TFIDFRetriever()
    dense_retriever = DenseRetriever(
        model_name=embedding_model
    )
    
    # Configure hybrid with both retrievers
    retriever_kwargs = {
        'retrievers': {
            'tfidf': tfidf_retriever,
            'dense': dense_retriever
        },
        'weights': {'tfidf': 0.3, 'dense': 0.7}
    }
    
    if use_reranking:
        retriever_type = 'rerank'
        retriever_kwargs = {
            'base_retriever_type': 'hybrid',
            'base_retriever_kwargs': retriever_kwargs,
            'cross_encoder_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'initial_top_k': 10
        }

retriever = get_retriever(retriever_type, **retriever_kwargs)

# Get generator
generator = get_generator(provider="together")

FILES_DIR = os.path.join(os.path.dirname(__file__), 'files')
COMPLETE_MENU_FILE = os.path.join(FILES_DIR, 'complete_menu.txt')

CONTEXT_DIR = os.path.join(os.path.dirname(__file__), 'context')
CONTEXT_FILE = os.path.join(CONTEXT_DIR, 'context.txt')
CONTEXT_URL = (
    'https://api.toteat.com/mw/or/1.0/products?'
    'xir=1787266874917892&xil=3&xiu=1002&xapitoken=uNyXp3DkfFVQgMbLeynFGvRJwk7fMp5d&activeProducts=true'
)

def update_context_file():
    os.makedirs(CONTEXT_DIR, exist_ok=True)
    os.makedirs(FILES_DIR, exist_ok=True)
    try:
        response = requests.get(CONTEXT_URL, timeout=30)
        response.raise_for_status()
        with open(CONTEXT_FILE, 'w', encoding='utf-8') as f:
            f.write(response.text)
        with open(COMPLETE_MENU_FILE, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"context.txt y complete_menu.txt actualizados exitosamente a las {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"Error actualizando context.txt y complete_menu.txt: {e}")

# Actualizar al iniciar el servidor ANTES de procesar documentos
update_context_file()

# Initialize document processing and indexing at application startup
documents = processor.load_documents()
chunks = processor.chunk_documents()
retriever.index_documents(chunks)
print(f"Indexed {len(chunks)} document chunks from {len(documents)} documents using {retriever_type} retriever")

# Programar actualización diaria a medianoche
scheduler = BackgroundScheduler()
scheduler.add_job(update_context_file, 'cron', hour=0, minute=0)
scheduler.start()

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for service monitoring."""
    return jsonify({"status": "healthy", "service": "rag-service"})

# RAG query endpoint - main endpoint for chatbot integration
@app.route("/api/rag/query", methods=["POST"])
def query_rag():
    """API endpoint for querying the RAG system from chatbot."""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Missing query parameter"}), 400
        
        query = data['query']
        top_k = data.get('top_k', 7)
        session_id = data.get('session_id')
        
        # Create a new session if one doesn't exist
        if not session_id:
            session_id = dialog_manager.create_session()
        
        # Get conversation history
        conversation_history = dialog_manager.format_history_for_llm(session_id, include_last_n_turns=3)
        
        # Retrieve relevant documents
        retrieved_docs = retriever.retrieve(query, top_k=top_k)
        
        # Generate response
        response_text = generator.generate_response(query, retrieved_docs, conversation_history)
        
        # Add to conversation history
        dialog_manager.add_to_history(session_id, "user", query)
        dialog_manager.add_to_history(session_id, "assistant", response_text)
        
        # Return response with session ID and metadata
        return jsonify({
            "success": True,
            "query": query,
            "response": response_text,
            "session_id": session_id,
            "retrieved_documents": retrieved_docs,
            "metadata": {
                "top_k": top_k,
                "retriever_type": retriever_type,
                "num_chunks_indexed": len(chunks)
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

# Endpoint to get conversation history
@app.route("/api/conversation/history", methods=["POST"])
def get_conversation_history():
    """API endpoint to get conversation history for a session."""
    try:
        data = request.json
        if not data or 'session_id' not in data:
            return jsonify({"error": "Missing session_id parameter"}), 400
        
        session_id = data['session_id']
        history = dialog_manager.get_history(session_id)
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "history": history
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

# Clear conversation history endpoint
@app.route("/api/conversation/clear", methods=["POST"])
def clear_conversation():
    """API endpoint to clear conversation history."""
    try:
        data = request.json
        if not data or 'session_id' not in data:
            return jsonify({"error": "Missing session_id parameter"}), 400
        
        session_id = data['session_id']
        dialog_manager.clear_history(session_id)
        
        return jsonify({
            "success": True,
            "message": "Conversation history cleared",
            "session_id": session_id
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

# Create new session endpoint
@app.route("/api/conversation/new", methods=["POST"])
def create_new_session():
    """API endpoint to create a new conversation session."""
    try:
        session_id = dialog_manager.create_session()
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "message": "New session created successfully"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

# Service information endpoint
@app.route("/api/info", methods=["GET"])
def service_info():
    """API endpoint to get service information and configuration."""
    return jsonify({
        "service": "RAG Service",
        "version": "1.0.0",
        "retriever_type": retriever_type,
        "embedding_model": embedding_model if retriever_type != 'tfidf' else None,
        "use_reranking": use_reranking,
        "documents_indexed": len(chunks),
        "total_documents": len(documents),
        "endpoints": {
            "query": "/api/rag/query",
            "new_session": "/api/conversation/new",
            "get_history": "/api/conversation/history",
            "clear_history": "/api/conversation/clear",
            "health": "/health",
            "info": "/api/info"
        }
    })

if __name__ == "__main__":
    # Run the service
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    print(f"Starting RAG Service on port {port}")
    print(f"Debug mode: {debug_mode}")
    print(f"Available endpoints:")
    print(f"  - POST /api/rag/query - Main RAG query endpoint")
    print(f"  - POST /api/conversation/new - Create new session")
    print(f"  - POST /api/conversation/history - Get conversation history")
    print(f"  - POST /api/conversation/clear - Clear conversation history")
    print(f"  - GET /health - Health check")
    print(f"  - GET /api/info - Service information")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
