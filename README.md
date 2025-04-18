# DotLLM

DotLLM is a vLLM wrapper that serves an OpenAI-compatible API using custom logit processors.

## Installation

```bash
# Install from source
pip install -e .
```

## Usage

```bash
# Start the API server with all vLLM OpenAI serve options
dotllm --model <model_id_or_path> --host 0.0.0.0 --port 8000

# Alternatively, you can use the serve subcommand (both approaches work the same)
dotllm serve --model <model_id_or_path> --host 0.0.0.0 --port 8000

# Use with OpenAI client
OPENAI_API_BASE=http://localhost:8000 OPENAI_API_KEY=dummy python -c "import openai; print(openai.ChatCompletion.create(model='mistralai/Mistral-7B-Instruct-v0.2', messages=[{'role': 'user', 'content': 'Hello!'}]))"
```

## CLI Options

DotLLM inherits all the CLI options from vLLM's OpenAI API server. You can use the same parameters:

```bash
dotllm --help
```

To see the version:

```bash
dotllm --version
```
