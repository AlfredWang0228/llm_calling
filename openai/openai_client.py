# openai_client.py

from openai import OpenAI
import yaml
import os
from typing import Dict, Any
import logging
from functools import wraps
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def logging_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        if len(args) > 1:  # Instance method call
            system_prompt = args[1]
            user_prompt = args[2]
        else:  # Direct function call
            system_prompt = kwargs.get('system_prompt', '')
            user_prompt = kwargs.get('user_prompt', '')
        
        input_word_count = len(system_prompt.split()) + len(user_prompt.split())
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        output_word_count = len(result.split())
        
        logging.info(f"Execution Time: {execution_time:.2f} seconds")
        logging.info(f"Input Word Count: {input_word_count}")
        logging.info(f"Output Word Count: {output_word_count}")
        return result
    
    return wrapper

class OpenAIClient:
    def __init__(self, config_path: str = "openai_config.yaml"):
        """
        Initialize OpenAIClient class and load configuration file.
        :param config_path: The path to the configuration file
        """
        self.config = self.load_config(config_path)
        self.client = OpenAI(api_key=self.config.get('api_key') or os.environ.get("OPENAI_API_KEY"))
        self.model = self.config['model']

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load the configuration file.
        :param config_path: The path to the configuration file
        :return: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            raise Exception(f"Failed to load config file: {str(e)}")

    @logging_decorator
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate response using the specified model.
        :param system_prompt: System prompt to set the behavior of the model
        :param user_prompt: User prompt, i.e., the question or task to be answered/completed
        :return: Generated response text
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.config.get('generation_params', {})
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return f"An error occurred: {str(e)}"

# Example usage
if __name__ == "__main__":
    openai_client = OpenAIClient()
    system_prompt = "You are a helpful assistant with expertise in programming and technology."
    user_prompt = "Explain the concept of recursion in programming."
    result = openai_client.generate_response(system_prompt, user_prompt)
    print(result)