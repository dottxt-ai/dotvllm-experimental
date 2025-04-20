# DotLLM

DotLLM is a vLLM wrapper that serves an OpenAI-compatible API using custom logit processors.

⚠️ This is an experimental project. Not for production use. ⚠️

## Optimizations

Here a few optimization ideas, that shouldn't be too hard to implement now that
we use a custom `AsyncEngine`:

- Run the compilation during the KV-cache computation and only schedule the request for generation once compilation is done. This way we're not blocking the generation and reduce the throughput.
- Compute the mask and load it on GPU during the forward pass.

## Experiments

Measuring the maximum throughput measured with `bench.py` with GPT2 on my machine:

- 62f8604: 3.20 req/s
- efd3441: 4.40 req/s (+37.5%)

## The sharp bits

- We use a `ProcessPool` instead of a `ThreadPool` to compile the indexes in parallel. As a result we need to serialize/deserialize the indexes, which incurs [a performance penalty](https://github.com/dottxt-ai/dotregex/issues/335).
- The server shuts down whenever the generation fails because of an error with structured generation. This should be easy to fix but I didn't get around to do it.

## Installation

```bash
# Install from source
pip install -e .
```

Which will also install vLLM.

## Usage

`dotllm` wraps vLLM, and it exposes the same OpenAI API. To run a server:

```bash
# Start the API server with all vLLM OpenAI serve options
dotllm --model <model_id_or_path> --host 0.0.0.0 --port 8000

# Use with OpenAI client
OPENAI_API_BASE=http://localhost:8000 OPENAI_API_KEY=dummy python -c "import openai; print(openai.ChatCompletion.create(model='mistralai/Mistral-7B-Instruct-v0.2', messages=[{'role': 'user', 'content': 'Hello!'}]))"
```

## CLI Options

DotLLM inherits all the CLI options from vLLM's OpenAI API server. You can use the same parameters:

```bash
dotllm --help
```
