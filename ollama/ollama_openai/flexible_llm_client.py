import os
import yaml
from openai import OpenAI
import ollama

class FlexibleLLMClient:
    def __init__(self, config_path='llm_config.yaml'):
        self.config = self.load_config(config_path)
        self.client = self.setup_client()

    def load_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    def setup_client(self):
        provider = self.config.get('provider', '').lower()
        if provider == 'ollama':
            return ollama
        elif provider == 'openai':
            return OpenAI(api_key=self.config.get('api_key'))
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_response(self, prompt, **kwargs):
        try:
            params = self.config.copy()
            params.update(kwargs)
            
            provider = params.pop('provider', '').lower()
            
            if provider == 'openai':
                messages = params.pop('messages', [])
                if not messages:
                    system_prompt = params.pop('system_prompt', "You are a helpful assistant.")
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    messages.append({"role": "user", "content": prompt})
                
                openai_params = ['model', 'frequency_penalty', 'presence_penalty', 'logit_bias', 
                                 'max_tokens', 'n', 'response_format', 'seed', 'stop', 'temperature', 
                                 'top_p', 'tools', 'tool_choice', 'user']
                api_params = {k: params[k] for k in openai_params if k in params}
                
                response = self.client.chat.completions.create(
                    messages=messages,
                    **api_params
                )
                return response.choices[0].message.content
            elif provider == 'ollama':
                model = params.pop('model', 'llama3.1')
                system = params.pop('system_prompt', "You are a helpful assistant.")
                
                response = self.client.generate(
                    model=model,
                    prompt=prompt,
                    system=system,
                    options=params
                )
                return response['response']
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            return f"Error generating response: {str(e)}"
