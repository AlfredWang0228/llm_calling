# Ollama Configuration Parameters Explained

This document provides detailed explanations for each parameter in the `ollama_config.yaml` file.

## model
The name of the model to be used. For example, "qwen2:7b" refers to the Qwen 2 model with 7 billion parameters.

## options

### temperature
- **Range**: 0.0 to 1.0
- **Purpose**: Controls the randomness of the output.
- **Effect**: 
  - Lower values (e.g., 0.2) make the output more focused and deterministic.
  - Higher values (e.g., 0.8) make the output more diverse and creative.
  - A value of 0.0 will always pick the most likely next token.

### top_p (nucleus sampling)
- **Range**: 0.0 to 1.0
- **Purpose**: Limits the cumulative probability of tokens to sample from.
- **Effect**:
  - Lower values (e.g., 0.5) make the output more focused and conservative.
  - Higher values (e.g., 0.9) allow for more diversity.
  - Often used as an alternative to temperature for controlling randomness.

### top_k
- **Range**: 1 to infinity (typically no more than 100)
- **Purpose**: Limits the number of tokens to sample from.
- **Effect**:
  - Lower values (e.g., 10) make the output more focused and deterministic.
  - Higher values (e.g., 50) allow for more diversity.
  - Often used in combination with top_p.

### num_ctx (context window size)
- **Typical values**: Powers of 2, such as 2048, 4096, 8192
- **Purpose**: Defines the size of the context window used to generate the next token.
- **Effect**:
  - Larger values allow the model to consider more previous text when generating new content.
  - Increases memory usage and computation time.

### num_predict (maximum number of tokens to generate)
- **Purpose**: Sets the maximum number of tokens the model will generate in a single response.
- **Effect**:
  - Limits the length of the generated text.
  - Can be used to control the verbosity of the model's responses.

### repeat_penalty
- **Range**: 1.0 to infinity
- **Purpose**: Penalizes repeated tokens.
- **Effect**:
  - Values greater than 1.0 discourage the model from repeating the same words or phrases.
  - Higher values (e.g., 1.5) make repetition less likely, potentially improving text diversity.

### presence_penalty
- **Range**: -2.0 to 2.0
- **Purpose**: Penalizes new tokens based on their presence in the entire generated text.
- **Effect**:
  - Positive values encourage the model to talk about new topics.
  - Negative values make the model more likely to stay on the same topic.

### frequency_penalty
- **Range**: -2.0 to 2.0
- **Purpose**: Penalizes new tokens based on their frequency in the entire generated text.
- **Effect**:
  - Positive values encourage the model to use a more diverse vocabulary.
  - Negative values make the model more likely to repeat common words.

### mirostat
- **Options**: 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0
- **Purpose**: Enables Mirostat sampling algorithm for adaptive control of perplexity.
- **Effect**:
  - Can help maintain a consistent level of randomness throughout the generated text.
  - Mirostat 2.0 (option 2) is generally more stable than the original Mirostat.

### mirostat_tau
- **Typical range**: 3.0 to 5.0
- **Purpose**: Sets the target entropy for Mirostat sampling.
- **Effect**:
  - Lower values lead to more focused and deterministic output.
  - Higher values allow for more randomness and diversity.

### mirostat_eta
- **Typical range**: 0.1 to 0.3
- **Purpose**: Sets the learning rate for Mirostat sampling.
- **Effect**:
  - Controls how quickly the algorithm adapts to reach the target entropy.
  - Higher values lead to faster adaptation but may cause instability.

### seed
- **Range**: Any integer value
- **Purpose**: Sets the random seed for reproducibility.
- **Effect**:
  - Using the same seed value will produce the same output for identical inputs.
  - Useful for debugging or when you need consistent results across multiple runs.

Remember that the optimal values for these parameters can vary depending on the specific task, model, and desired output characteristics. Experimentation is often necessary to find the best configuration for your use case.