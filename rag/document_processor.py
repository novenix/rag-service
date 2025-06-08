import os
import re
from typing import List, Dict

class DocumentProcessor:
    def __init__(self, documents_dir: str):
        """Initialize with the directory containing documents to process."""
        self.documents_dir = documents_dir
        self.documents = {}
        
    def load_documents(self) -> Dict[str, str]:
        """Load all text files from the documents directory."""
        for filename in os.listdir(self.documents_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.documents_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Remove filepath comment if present
                    content = re.sub(r'// filepath:.*\n', '', content)
                    self.documents[filename] = content
        return self.documents
    
    def chunk_documents(self, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of document chunks with metadata
        """
        chunks = []
        
        for doc_id, content in self.documents.items():
            if len(content) <= chunk_size:
                chunks.append({
                    "text": content,
                    "metadata": {"source": doc_id}
                })
            else:
                # Split longer documents
                for i in range(0, len(content), chunk_size - overlap):
                    chunk_text = content[i:i + chunk_size]
                    if len(chunk_text) < 100:  # Skip very small chunks
                        continue
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "source": doc_id,
                            "chunk_id": i // (chunk_size - overlap)
                        }
                    })
        
        return chunks
