# LLM Project Utils

This project demonstrates various methods to invoke large language models (LLMs), each with a corresponding YAML configuration file.

## Methods

### HuggingFace

- Configuration: `huggingface_mode/hf_config.yaml`
- Client: `huggingface_mode/huggingface_llm_client.py`

### LLaMA-CPP

- Configuration: `llama_cpp_mode/llama_config.yaml`
- Client: `llama_cpp_mode/llama_cpp_client.py`

### Ollama

- Configuration: `ollama/ollama_config.yaml`
- Client: `ollama/ollama_client.py`

### OpenAI

- Configuration: `openai/openai_config.yaml`
- Client: `openai/openai_client.py`

Each method uses a specific YAML configuration file to set up and call the respective LLM.
