import os
import openai

# Set the API base URL to your DotLLM server
# Default is http://localhost:8000
DOTLLM_API_URL = "http://0.0.0.0:8000/v1"

# API key can be any string (DotLLM doesn't validate it)
client = openai.OpenAI(
    base_url=DOTLLM_API_URL,
    api_key="dummy"
)

# Send a completion request
response = client.completions.create(
    model="gpt2",  # Use the model name provided to DotLLM
    prompt="What is the capital of France?",
    max_tokens=100,
    temperature=0.7
)

# Print the response
print(response.choices[0].text)
