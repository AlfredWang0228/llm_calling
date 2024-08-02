# huggingface_llm_client.py

import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from functools import wraps
import time
import logging
import yaml
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def logging_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Correctly handle args whether the method is called on an instance or not
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

class HuggingFaceLLMClient:
    def __init__(self, config_path='hf_config.yaml'):
        self.config = self.load_config(config_path)
        self.tokenizer, self.model = self.setup_model()
        self.pipeline = self.setup_pipeline()

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def setup_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_id'], token=self.config['token'])
        
        model_kwargs = self.config['model_kwargs'].copy()
        
        # Check if quantization is enabled
        if self.config.get('use_quantization', False):
            from transformers import BitsAndBytesConfig
            quant_config = self.config.get('quantization_config', {})
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=quant_config.get('load_in_4bit', True),
                bnb_4bit_compute_dtype=getattr(torch, quant_config.get('bnb_4bit_compute_dtype', 'float16')),
                bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
                bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4')
            )
            model_kwargs['quantization_config'] = quantization_config
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            self.config['model_id'],
            token=self.config['token'],
            device_map=self.config['device'],
            **model_kwargs
        )
        
        return tokenizer, model

    def setup_pipeline(self):
        # Check if the model is distributed across devices
        hf_device_map = getattr(self.model, "hf_device_map", None)
        
        pipeline_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
        }
        
        # Only add the device argument if the model is not distributed
        if hf_device_map is None:
            pipeline_kwargs["device"] = self.config['device']
        
        return pipeline(
            "text-generation",
            **pipeline_kwargs
        )

    @logging_decorator
    def generate_text(self, system_prompt, user_prompt, print_prompt=False):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        if print_prompt:
            print("Generated Prompt: ", prompt)
        
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.config['max_new_tokens'],
            eos_token_id=terminators,
            do_sample=self.config['do_sample'],
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        
        return outputs[0]["generated_text"][len(prompt):]

# Usage example
if __name__ == "__main__":
    client = HuggingFaceLLMClient()
    
    system_prompt = "You're a helpful assistant."
    user_prompt = "What is the architecture of LLaMa-3?"
    
    answer = client.generate_text(system_prompt, user_prompt)
    print(answer)

    # If you want to display as Markdown, you can use a library like IPython
    # from IPython.display import display, Markdown
    # display(Markdown(answer))