# llama_cpp_client.py

import logging
import time
from functools import wraps
from llama_cpp import Llama
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def logging_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        system_prompt = kwargs.get('system_prompt', '')
        user_prompt = kwargs.get('user_prompt', '')
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            raise ValueError("System and user prompts must be strings.")
        
        input_word_count = len(system_prompt.split()) + len(user_prompt.split())
        result = func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        output_word_count = len(result.split()) if result else 0
        logging.info(f"Execution Time: {execution_time:.2f} seconds")
        logging.info(f"Input Word Count: {input_word_count}")
        logging.info(f"Output Word Count: {output_word_count}")
        return result
    
    return wrapper

class LlamaCppClient:
    def __init__(self, config_path='llama_config.yaml'):
        self.config = self.load_config(config_path)
        self.model = self.load_model()

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def load_model(self):
        model_params = {
            'model_path': self.config['model_path'],
            'verbose': self.config.get('verbose', False),
            'n_gpu_layers': self.config.get('n_gpu_layers', 0),
            'n_ctx': self.config.get('n_ctx', 4096),
            'main_gpu': self.config.get('main_gpu', 0),
            'tensor_split': self.config.get('tensor_split'),
            'offload_kqv': self.config.get('offload_kqv', False),
            'flash_attn': self.config.get('flash_attn', False)
        }
        return Llama(**model_params)

    @logging_decorator
    def generate_response(self, system_prompt, user_prompt, max_tokens=None, seed=None):
        max_tokens = max_tokens or self.config.get('max_tokens', 8192)
        seed = seed or self.config.get('seed', 42)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            output = self.model.create_chat_completion(
                messages=messages,
                stop=["<|eot_id|>", "<|end_of_text|>"],
                max_tokens=max_tokens,
                temperature=self.config.get('temperature', 0.0),
                seed=seed
            )["choices"][0]["message"]["content"]
            return output
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            return None

# Usage example
if __name__ == "__main__":
    client = LlamaCppClient()
    
    system_prompt = "You are a helpful assistant."
    user_prompt = "Explain the concept of machine learning in simple terms."
    
    response = client.generate_response(system_prompt, user_prompt)
    print(f"System Prompt: {system_prompt}")
    print(f"User Prompt: {user_prompt}")
    print(f"Response: {response}")