import sys
import os
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluation import RAGEvaluator
from rag.retriever import get_retriever  # Use the factory function instead
from rag.generator import get_generator
from rag.document_processor import DocumentProcessor
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run RAG system evaluation')
    parser.add_argument('--retriever', type=str, 
                      choices=['tfidf', 'dense', 'hybrid', 'rerank'], 
                      default='tfidf',
                      help='Retriever type to use (tfidf, dense, hybrid, rerank)')
    parser.add_argument('--test_file', type=str, 
                      default='evaluation/test_queries.json',
                      help='Path to test queries JSON file')
    parser.add_argument('--embedding_model', type=str,
                      default='all-MiniLM-L6-v2',
                      help='Embedding model for dense/hybrid retrievers')
    args = parser.parse_args()
    
    # Initialize retriever based on type using the factory function
    if args.retriever == 'hybrid':
        # For hybrid, we need to create both base retrievers
        tfidf_retriever = get_retriever('tfidf')
        dense_retriever = get_retriever('dense', model_name=args.embedding_model)
        retriever = get_retriever('hybrid', retrievers={
            'tfidf': tfidf_retriever,
            'dense': dense_retriever
        })
    elif args.retriever == 'rerank':
        # For reranking, we create a base retriever (dense) and wrap it
        base_retriever = get_retriever('dense', model_name=args.embedding_model)
        retriever = get_retriever('rerank', base_retriever=base_retriever)
    else:
        # For simple retrievers (tfidf, dense)
        retriever_kwargs = {}
        if args.retriever == 'dense':
            retriever_kwargs['model_name'] = args.embedding_model
        retriever = get_retriever(args.retriever, **retriever_kwargs)
    
    # Load and index documents before evaluation
    documents_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'files')
    processor = DocumentProcessor(documents_dir)
    documents = processor.load_documents()
    chunks = processor.chunk_documents()
    retriever.index_documents(chunks)
    print(f"Indexed {len(chunks)} document chunks from {len(documents)} documents")
    
    generator = get_generator(provider="together")
    
    # Initialize and run evaluator
    evaluator = RAGEvaluator(retriever, generator)
    results = evaluator.run_evaluation(args.test_file)
    evaluator.print_evaluation_report(results)
    
    # Save detailed results to file
    with open(f'evaluation_results_{args.retriever}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to evaluation_results_{args.retriever}.json")

if __name__ == "__main__":
    main()
