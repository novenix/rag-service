import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
import json
import os

class RAGEvaluator:
    def __init__(self, retriever, generator):
        """
        Initialize the evaluator with retriever and generator components.
        
        Args:
            retriever: Document retrieval component
            generator: Response generation component
        """
        self.retriever = retriever
        self.generator = generator
        self.rouge = Rouge()
        
    def load_test_queries(self, test_file):
        """
        Load test queries and expected responses from a JSON file.
        
        Args:
            test_file: Path to the JSON file with test data
            
        Returns:
            List of test query objects
        """
        with open(test_file, 'r') as f:
            return json.load(f)
    
    def evaluate_retrieval(self, query, relevant_docs, top_k=3):
        """
        Evaluate the retrieval component using precision and recall.
        
        Args:
            query: Query string
            relevant_docs: List of relevant document identifiers (source filename)
            top_k: Number of documents to retrieve
            
        Returns:
            Dict with precision and recall values
        """
        retrieved_docs = self.retriever.retrieve(query, top_k)
        
        # Extract document identifiers from retrieved documents
        # Using source filename as the primary identifier
        retrieved_sources = [doc.get('metadata', {}).get('source', '') for doc in retrieved_docs]
        
        # Calculate precision and recall
        true_positives = len(set(retrieved_sources) & set(relevant_docs))
        precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
        recall = true_positives / len(relevant_docs) if relevant_docs else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'retrieved_docs': retrieved_docs
        }
    
    def evaluate_generation(self, query, retrieved_docs, expected_response):
        """
        Evaluate the generation component using ROUGE score.
        
        Args:
            query: Query string
            retrieved_docs: Retrieved documents
            expected_response: Expected response string
            
        Returns:
            Dict with generation metrics
        """
        # Use generate_response instead of generate to match the generator interface
        generated_response = self.generator.generate_response(query, retrieved_docs)
        
        # Calculate ROUGE scores
        try:
            rouge_scores = self.rouge.get_scores(generated_response, expected_response)[0]
        except Exception as e:
            print(f"ROUGE calculation failed: {e}")
            # Fallback if ROUGE fails
            rouge_scores = {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}
        
        return {
            'generated_response': generated_response,
            'rouge_1': rouge_scores['rouge-1']['f'],
            'rouge_2': rouge_scores['rouge-2']['f'],
            'rouge_l': rouge_scores['rouge-l']['f']
        }
    
    def run_evaluation(self, test_file):
        """
        Run complete evaluation on test queries.
        
        Args:
            test_file: Path to test data file
            
        Returns:
            Dict with evaluation results
        """
        test_queries = self.load_test_queries(test_file)
        results = []
        
        retrieval_metrics = {'precision': [], 'recall': [], 'f1': []}
        generation_metrics = {'rouge_1': [], 'rouge_2': [], 'rouge_l': []}
        
        for test_case in test_queries:
            query = test_case['query']
            relevant_docs = test_case.get('relevant_docs', [])
            expected_response = test_case['expected_response']
            
            # Evaluate retrieval
            retrieval_result = self.evaluate_retrieval(query, relevant_docs)
            
            # Evaluate generation
            generation_result = self.evaluate_generation(
                query, 
                retrieval_result['retrieved_docs'], 
                expected_response
            )
            
            # Collect metrics
            for key in retrieval_metrics:
                retrieval_metrics[key].append(retrieval_result[key])
                
            for key in generation_metrics:
                generation_metrics[key].append(generation_result[key])
            
            # Store individual results
            results.append({
                'query': query,
                'retrieval': retrieval_result,
                'generation': generation_result
            })
        
        # Calculate averages
        avg_retrieval = {k: np.mean(v) for k, v in retrieval_metrics.items()}
        avg_generation = {k: np.mean(v) for k, v in generation_metrics.items()}
        
        return {
            'individual_results': results,
            'average_retrieval': avg_retrieval,
            'average_generation': avg_generation
        }
    
    def print_evaluation_report(self, results):
        """
        Print a formatted evaluation report.
        
        Args:
            results: Evaluation results from run_evaluation
        """
        print("\n===== RAG Evaluation Report =====")
        print("\nRetrieval Performance:")
        print(f"Average Precision: {results['average_retrieval']['precision']:.4f}")
        print(f"Average Recall: {results['average_retrieval']['recall']:.4f}")
        print(f"Average F1: {results['average_retrieval']['f1']:.4f}")
        
        print("\nGeneration Performance:")
        print(f"Average ROUGE-1: {results['average_generation']['rouge_1']:.4f}")
        print(f"Average ROUGE-2: {results['average_generation']['rouge_2']:.4f}")
        print(f"Average ROUGE-L: {results['average_generation']['rouge_l']:.4f}")
        
        print("\nIndividual Query Results:")
        for i, result in enumerate(results['individual_results']):
            print(f"\nQuery {i+1}: {result['query']}")
            print(f"  Retrieval: P={result['retrieval']['precision']:.2f}, " +
                  f"R={result['retrieval']['recall']:.2f}, F1={result['retrieval']['f1']:.2f}")
            
            # Display retrieved documents
            print("  Retrieved documents:")
            for j, doc in enumerate(result['retrieval']['retrieved_docs'][:3]):  # Show top 3
                source = doc.get('metadata', {}).get('source', 'unknown')
                score = doc.get('score', 0.0)
                print(f"    {j+1}. {source} (score: {score:.2f})")
            
            print(f"  Generation: ROUGE-L={result['generation']['rouge_l']:.2f}")
            print(f"  Response: {result['generation']['generated_response'][:100]}...")
