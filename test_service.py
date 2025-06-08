#!/usr/bin/env python3
"""
Script de prueba para el servicio RAG.
Prueba los endpoints principales del servicio.
"""

import requests
import json
import time

# Configuración del servicio
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Prueba el endpoint de health check."""
    print("🔍 Probando health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check exitoso: {data}")
            return True
        else:
            print(f"❌ Health check falló: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en health check: {e}")
        return False

def test_service_info():
    """Prueba el endpoint de información del servicio."""
    print("\n🔍 Probando información del servicio...")
    try:
        response = requests.get(f"{BASE_URL}/api/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Info del servicio:")
            print(f"   - Tipo de retriever: {data.get('retriever_type')}")
            print(f"   - Documentos indexados: {data.get('documents_indexed')}")
            print(f"   - Total documentos: {data.get('total_documents')}")
            return True
        else:
            print(f"❌ Info del servicio falló: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en info del servicio: {e}")
        return False

def test_new_session():
    """Prueba la creación de una nueva sesión."""
    print("\n🔍 Probando creación de sesión...")
    try:
        response = requests.post(f"{BASE_URL}/api/conversation/new", 
                               headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            data = response.json()
            session_id = data.get('session_id')
            print(f"✅ Nueva sesión creada: {session_id}")
            return session_id
        else:
            print(f"❌ Creación de sesión falló: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error en creación de sesión: {e}")
        return None

def test_rag_query(session_id=None):
    """Prueba una consulta RAG."""
    print(f"\n🔍 Probando consulta RAG (sesión: {session_id})...")
    
    test_query = "¿Cuál es la misión de HistoriaCard?"
    
    payload = {
        "query": test_query,
        "top_k": 3
    }
    
    if session_id:
        payload["session_id"] = session_id
    
    try:
        response = requests.post(f"{BASE_URL}/api/rag/query",
                               headers={'Content-Type': 'application/json'},
                               json=payload)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Consulta RAG exitosa:")
                print(f"   - Pregunta: {data.get('query')}")
                print(f"   - Respuesta: {data.get('response')[:100]}...")
                print(f"   - Session ID: {data.get('session_id')}")
                print(f"   - Documentos recuperados: {len(data.get('retrieved_documents', []))}")
                return data.get('session_id')
            else:
                print(f"❌ Consulta RAG falló: {data.get('error')}")
                return None
        else:
            print(f"❌ Consulta RAG falló: {response.status_code}")
            print(f"   Respuesta: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error en consulta RAG: {e}")
        return None

def test_conversation_history(session_id):
    """Prueba obtener el historial de conversación."""
    if not session_id:
        print("\n⚠️  No hay session_id para probar historial")
        return False
        
    print(f"\n🔍 Probando historial de conversación (sesión: {session_id})...")
    
    try:
        response = requests.post(f"{BASE_URL}/api/conversation/history",
                               headers={'Content-Type': 'application/json'},
                               json={"session_id": session_id})
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                history = data.get('history', [])
                print(f"✅ Historial obtenido: {len(history)} mensajes")
                for i, msg in enumerate(history):
                    print(f"   {i+1}. {msg.get('role')}: {msg.get('content')[:50]}...")
                return True
            else:
                print(f"❌ Obtener historial falló: {data.get('error')}")
                return False
        else:
            print(f"❌ Obtener historial falló: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en obtener historial: {e}")
        return False

def main():
    """Ejecuta todas las pruebas."""
    print("🚀 Iniciando pruebas del servicio RAG...")
    print("=" * 50)
    
    # Prueba 1: Health check
    if not test_health_check():
        print("\n❌ El servicio no está disponible. Asegúrate de que esté ejecutándose en http://localhost:5000")
        return
    
    # Prueba 2: Info del servicio
    test_service_info()
    
    # Prueba 3: Nueva sesión
    session_id = test_new_session()
    
    # Prueba 4: Consulta RAG
    session_id = test_rag_query(session_id)
    
    # Prueba 5: Segunda consulta para probar historial
    if session_id:
        print(f"\n🔍 Probando segunda consulta en la misma sesión...")
        test_rag_query(session_id)
    
    # Prueba 6: Historial de conversación
    test_conversation_history(session_id)
    
    print("\n" + "=" * 50)
    print("🎉 Pruebas completadas!")

if __name__ == "__main__":
    main()