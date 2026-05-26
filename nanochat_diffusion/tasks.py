"""
Evaluation tasks for diffusion LLM.

Provides standardized tasks for evaluating diffusion model capabilities:
- GSM8K: Grade school math word problems
- MMLU: Multiple choice across 57 subjects
- ARC: Alphabetical reasoning challenges
- HumanEval: Simple coding tasks
- Custom JSON: Custom task from JSONL format
"""

import torch
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class TaskExample:
    """A single evaluation example."""
    id: str
    question: str
    options: Optional[List[str]] = None
    answer: Optional[str] = None
    prompt: str = ""
    completion: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TaskMixture:
    """
    Mix multiple tasks with given weights.
    
    Example:
        task = TaskMixture({
            'gsm8k': 0.7,
            'mmlu': 0.3,
        })
    """
    
    def __init__(self, task_dict: Dict[str, float]):
        self.task_dict = task_dict
        self.tasks = {}
        self.examples = []
        
        for task_name, weight in task_dict.items():
            if task_name == 'gsm8k':
                self.tasks['gsm8k'] = GSM8KTask()
            elif task_name == 'mmlu':
                self.tasks['mmlu'] = MMLUTask()
            elif task_name == 'arc':
                self.tasks['arc'] = ARCTask()
            elif task_name == 'humaneval':
                self.tasks['humaneval'] = HumanEvalTask()
            elif task_name == 'customjson':
                self.tasks['customjson'] = CustomJSONTask()
            else:
                raise ValueError(f"Unknown task: {task_name}")
    
    def get_examples(self, n: int = 100) -> List[TaskExample]:
        """Get N examples from the task mixture."""
        examples = []
        
        # Get equal number from each task
        per_task = n // len(self.task_dict)
        
        for task_name, task in self.tasks.items():
            task_examples = task.get_examples(per_task)
            examples.extend(task_examples)
        
        # Shuffle
        random.shuffle(examples)
        return examples[:n]
    
    def get_weights(self) -> Dict[str, float]:
        """Get task weights."""
        return self.task_dict


class GSM8KTask:
    """Grade School Math 8K task."""
    
    def __init__(self):
        self.name = "gsm8k"
        self.examples = []
        self._load_data()
    
    def _load_data(self):
        """Load GSM8K data (using sample for demo)."""
        self.examples = [
            TaskExample(
                id="gsm8k_0",
                question="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                answer="72",
                metadata={"type": "math"}
            ),
            TaskExample(
                id="gsm8k_1",
                question="Winston has a 1 dollar coin. He goes to the fruit vendor and buys 4 oranges for 25 cents each. How much change does Winston get back?",
                answer="75",
                metadata={"type": "math"}
            ),
            TaskExample(
                id="gsm8k_2",
                question="Wilma's garden is 3 rows by 4 rows. She has 3 roses per plant. How many roses are in her garden?",
                answer="36",
                metadata={"type": "math"}
            ),
        ]
    
    def get_examples(self, n: int = 100) -> List[TaskExample]:
        return self.examples[:n]
    
    def format_prompt(self, example: TaskExample) -> str:
        return f"Question: {example.question}\nAnswer:"
    
    def compute_score(self, generated: str, answer: str) -> float:
        """Compute score for generated answer."""
        return 1.0 if generated.strip() == answer.strip() else 0.0


class MMLUTask:
    """Multiple Choice Question task."""
    
    def __init__(self):
        self.name = "mmlu"
        self.examples = []
        self._load_data()
    
    def _load_data(self):
        """Load MMLU data (using sample for demo)."""
        self.examples = [
            TaskExample(
                id="mmlu_0",
                question="Which of the following is a characteristic of a firm in monopolistic competition?",
                options=["A", "B", "C", "D"],
                answer="B",
                metadata={"type": "multiple_choice", "subject": "economics"}
            ),
            TaskExample(
                id="mmlu_1",
                question="What is the primary function of the mitochondria in a cell?",
                options=["A. Protein synthesis", "B. Energy production", "C. DNA replication", "D. Cell division"],
                answer="B",
                metadata={"type": "multiple_choice", "subject": "biology"}
            ),
        ]
    
    def get_examples(self, n: int = 100) -> List[TaskExample]:
        return self.examples[:n]
    
    def format_prompt(self, example: TaskExample) -> str:
        prompt = f"Question: {example.question}\nOptions:"
        for opt in example.options or []:
            prompt += f"\n{opt}"
        prompt += "\nAnswer:"
        return prompt
    
    def compute_score(self, generated: str, answer: str) -> float:
        """Compute score for multiple choice."""
        return 1.0 if generated.strip().startswith(answer) else 0.0


class ARCTask:
    """Alphabetical Reasoning Challenge task."""
    
    def __init__(self):
        self.name = "arc"
        self.examples = []
        self._load_data()
    
    def _load_data(self):
        """Load ARC data (using sample for demo)."""
        self.examples = [
            TaskExample(
                id="arc_0",
                question="If all Bloops are Razzies and all Razzies are Lazzies, then all Bloops are definitely Lazzies.",
                options=["Yes", "No"],
                answer="Yes",
                metadata={"type": "logical_reasoning"}
            ),
            TaskExample(
                id="arc_1",
                question="A car travels at 60 miles per hour. How far will it travel in 2.5 hours?",
                options=["100 miles", "120 miles", "150 miles", "180 miles"],
                answer="150 miles",
                metadata={"type": "math"}
            ),
        ]
    
    def get_examples(self, n: int = 100) -> List[TaskExample]:
        return self.examples[:n]
    
    def format_prompt(self, example: TaskExample) -> str:
        prompt = f"Question: {example.question}"
        if example.options:
            for opt in example.options:
                prompt += f"\n{opt}"
        prompt += "\nAnswer:"
        return prompt
    
    def compute_score(self, generated: str, answer: str) -> float:
        return 1.0 if generated.strip() == answer.strip() else 0.0


class HumanEvalTask:
    """Simple coding task."""
    
    def __init__(self):
        self.name = "humaneval"
        self.examples = []
        self._load_data()
    
    def _load_data(self):
        """Load HumanEval data (using sample for demo)."""
        self.examples = [
            TaskExample(
                id="humaneval_0",
                question="Write a Python function to find the maximum of two numbers.",
                answer="def max_two(a, b):\n    return a if a > b else b",
                metadata={"type": "coding", "language": "python"}
            ),
            TaskExample(
                id="humaneval_1",
                question="Write a Python function to check if a string is a palindrome.",
                answer="def is_palindrome(s):\n    return s == s[::-1]",
                metadata={"type": "coding", "language": "python"}
            ),
        ]
    
    def get_examples(self, n: int = 100) -> List[TaskExample]:
        return self.examples[:n]
    
    def format_prompt(self, example: TaskExample) -> str:
        prompt = f"Question: {example.question}\nCode:"
        return prompt
    
    def compute_score(self, generated: str, answer: str) -> float:
        return 1.0 if generated.strip() == answer.strip() else 0.0


class CustomJSONTask:
    """Custom task from JSONL format."""
    
    def __init__(self, jsonl_path: str = ""):
        self.name = "customjson"
        self.examples = []
        self.jsonl_path = jsonl_path
        if jsonl_path:
            self._load_from_jsonl()
    
    def _load_from_jsonl(self):
        """Load examples from JSONL file."""
        self.examples = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                example = TaskExample(
                    id=data.get('id', ''),
                    question=data.get('question', ''),
                    options=data.get('options', []),
                    answer=data.get('answer', ''),
                    prompt=data.get('prompt', ''),
                    completion=data.get('completion', ''),
                    metadata=data.get('metadata', {})
                )
                self.examples.append(example)
    
    def get_examples(self, n: int = 100) -> List[TaskExample]:
        return self.examples[:n]
    
    def format_prompt(self, example: TaskExample) -> str:
        if example.prompt:
            return example.prompt
        return f"Question: {example.question}\nAnswer:"
    
    def compute_score(self, generated: str, answer: str) -> float:
        return 1.0 if generated.strip() == answer.strip() else 0.0


def evaluate_model(model, task: TaskMixture, num_examples: int = 100, max_tokens: int = 128):
    """
    Evaluate a model on a task mixture.
    
    Args:
        model: The model to evaluate
        task: Task mixture to evaluate on
        num_examples: Number of examples to evaluate on
        max_tokens: Maximum tokens to generate
    
    Returns:
        Evaluation metrics
    """
    from nanochat_diffusion.diffusion_sampler import DiffusionSampler
    
    examples = task.get_examples(num_examples)
    metrics = {
        'total': 0,
        'correct': 0,
        'scores': [],
        'details': []
    }
    
    sampler = DiffusionSampler(model)
    
    for example in examples[:num_examples]:
        prompt = task.format_prompt(example)
        
        # Generate response
        generated = sampler.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.8
        )
        
        # Compute score
        score = task.compute_score(generated, example.answer)
        
        metrics['total'] += 1
        metrics['correct'] += score
        metrics['scores'].append(score)
        
        metrics['details'].append({
            'id': example.id,
            'question': example.question[:50],
            'generated': generated[:50],
            'score': score
        })
    
    metrics['accuracy'] = metrics['correct'] / max(1, metrics['total'])
    metrics['mean_score'] = sum(metrics['scores']) / max(1, len(metrics['scores']))
    
    return metrics
