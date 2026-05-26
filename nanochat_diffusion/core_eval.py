"""
Evaluate core model capabilities using DCLM-style evaluation.

Implements the DCLM (Decomposed Contextual Language Model) CORE metric
for evaluating language model capabilities.
"""

import torch
import torch.nn.functional as F
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Result of an evaluation."""
    model_name: str
    task_name: str
    score: float
    details: Dict = None
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'task_name': self.task_name,
            'score': self.score,
            'details': self.details or {},
            'metadata': self.metadata or {}
        }


class COREEvaluator:
    """
    Core evaluator for diffusion LLM.
    
    Evaluates model capabilities using DCLM-style metrics.
    """
    
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.results = []
    
    def evaluate_gsm8k(self, test_data: List[Dict], num_examples: int = 100) -> EvaluationResult:
        """
        Evaluate on GSM8K (Grade School Math).
        
        Args:
            test_data: List of dictionaries with 'question' and 'answer' keys
            num_examples: Number of examples to evaluate
        
        Returns:
            EvaluationResult with accuracy
        """
        correct = 0
        total = min(num_examples, len(test_data))
        
        for i, example in enumerate(test_data[:total]):
            # Generate answer
            generated = self._generate_answer(example['question'])
            
            # Check if answer is correct
            if self._is_correct(generated, example['answer']):
                correct += 1
        
        accuracy = correct / max(1, total)
        return EvaluationResult(
            model_name='diffusion_model',
            task_name='gsm8k',
            score=accuracy,
            details={'correct': correct, 'total': total}
        )
    
    def evaluate_mmlu(self, test_data: List[Dict], num_examples: int = 100) -> EvaluationResult:
        """
        Evaluate on MMLU (Multiple Choice QA).
        
        Args:
            test_data: List of dictionaries with 'question', 'options', 'answer' keys
            num_examples: Number of examples to evaluate
        
        Returns:
            EvaluationResult with accuracy
        """
        correct = 0
        total = min(num_examples, len(test_data))
        
        for i, example in enumerate(test_data[:total]):
            generated = self._generate_multiple_choice(example['question'], example['options'])
            
            if generated == example['answer']:
                correct += 1
        
        accuracy = correct / max(1, total)
        return EvaluationResult(
            model_name='diffusion_model',
            task_name='mmlu',
            score=accuracy,
            details={'correct': correct, 'total': total}
        )
    
    def evaluate_arc(self, test_data: List[Dict], num_examples: int = 100) -> EvaluationResult:
        """
        Evaluate on ARC (Alphabetical Reasoning Challenge).
        
        Args:
            test_data: List of dictionaries with 'question', 'options', 'answer' keys
            num_examples: Number of examples to evaluate
        
        Returns:
            EvaluationResult with accuracy
        """
        correct = 0
        total = min(num_examples, len(test_data))
        
        for i, example in enumerate(test_data[:total]):
            generated = self._generate_multiple_choice(example['question'], example['options'])
            
            if generated == example['answer']:
                correct += 1
        
        accuracy = correct / max(1, total)
        return EvaluationResult(
            model_name='diffusion_model',
            task_name='arc',
            score=accuracy,
            details={'correct': correct, 'total': total}
        )
    
    def evaluate_bits_per_byte(self, test_data: List[str], num_examples: int = 100) -> EvaluationResult:
        """
        Evaluate bits per byte (BPB) metric.
        
        Lower BPB indicates better compression/prediction.
        """
        total_bits = 0
        total_bytes = 0
        
        for i, text in enumerate(test_data[:num_examples]):
            # Encode text to bytes
            text_bytes = text.encode('utf-8')
            total_bytes += len(text_bytes)
            
            # Encode tokens
            tokens = self.tokenizer.encode(text) if self.tokenizer else [ord(c) for c in text]
            
            # Forward pass
            with torch.no_grad():
                input_tokens = torch.tensor(tokens, dtype=torch.long, device=self.model.get_device())
                logits = self.model(input_tokens.unsqueeze(0))
            
            # Compute cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_tokens.view(-1),
                ignore_index=0
            )
            
            # Convert to bits per byte
            bits_per_token = loss.item() / np.log(2)
            total_bits += bits_per_token * len(tokens)
        
        bpb = total_bits / max(1, total_bytes)
        return EvaluationResult(
            model_name='diffusion_model',
            task_name='bits_per_byte',
            score=bpb,
            details={'total_bits': total_bits, 'total_bytes': total_bytes}
        )
    
    def _generate_answer(self, question: str) -> str:
        """Generate an answer for a question."""
        # This is a simplified implementation
        # In practice, this would call the model's generation method
        prompt = f"Question: {question}\nAnswer:"
        # Simulate generation
        return "42"  # Placeholder
    
    def _generate_multiple_choice(self, question: str, options: List[str]) -> str:
        """Generate a multiple choice answer."""
        # Simplified implementation
        return options[0] if options else "A"
    
    def _is_correct(self, generated: str, expected: str) -> bool:
        """Check if generated answer is correct."""
        return generated.strip() == expected.strip()
    
    def run_all_evaluations(self, test_data: Dict[str, List[Dict]], num_examples: int = 100) -> List[EvaluationResult]:
        """
        Run all evaluations.
        
        Args:
            test_data: Dictionary with task names as keys and test data as values
            num_examples: Number of examples per task
        
        Returns:
            List of EvaluationResults
        """
        results = []
        
        if 'gsm8k' in test_data:
            results.append(self.evaluate_gsm8k(test_data['gsm8k'], num_examples))
        
        if 'mmlu' in test_data:
            results.append(self.evaluate_mmlu(test_data['mmlu'], num_examples))
        
        if 'arc' in test_data:
            results.append(self.evaluate_arc(test_data['arc'], num_examples))
        
        return results
    
    def get_results(self) -> List[Dict]:
        """Get all results as dictionaries."""
        return [r.to_dict() for r in self.results]
    
    def save_results(self, filepath: str = "core_eval_results.json"):
        """Save results to file."""
        with open(filepath, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)


def evaluate_model(model, tokenizer=None, test_data=None, task_name="gsm8k", num_examples=100):
    """
    Convenience function for evaluation.
    
    Args:
        model: The model to evaluate
        tokenizer: Optional tokenizer
        test_data: Test data for the task
        task_name: Name of the task
        num_examples: Number of examples to evaluate
    
    Returns:
        EvaluationResult
    """
    evaluator = COREEvaluator(model, tokenizer)
    
    if task_name == 'gsm8k':
        return evaluator.evaluate_gsm8k(test_data, num_examples)
    elif task_name == 'mmlu':
        return evaluator.evaluate_mmlu(test_data, num_examples)
    elif task_name == 'arc':
        return evaluator.evaluate_arc(test_data, num_examples)
    elif task_name == 'bits_per_byte':
        return evaluator.evaluate_bits_per_byte(test_data, num_examples)
    else:
        raise ValueError(f"Unknown task: {task_name}")
