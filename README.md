# DotLLM

DotLLM is a vLLM wrapper that serves an OpenAI-compatible API using custom logit processors.

## Optimizations

We can implement the following optimizations:

- [ ] Use a `ProcessPoolExecutor` instead of a `ThreadPoolExecutor` to compile indexes in parallel. To do this we'd need to serialize/deserialize the index, which might add a performance penalty.
- [ ] Run the compilation during the KV-cache computation and only schedule the request for generation once compilation is done. This way we're not blocking the generation and reduce the throughput.
- [ ] Compute the mask and load it on GPU during the forward pass.

We can also:
- [ ] Add a simple `Cache` implementation
- [ ] Implement a `CompilationManager` that uses a different instance to compile the index, and fetch it from remote storage (with local cache)
- [ ] Give selective multiplication a try.
- [ ] Modify the implementation so coalescence can be easily added.

We also need to add a benchmark script with multiple schemas and concurrent requests so we can make sure the optimizations are actually making a difference, and quantify them.

## Experiments

Measuring the maximum throughput measured with `bench.py` with GPT2 on my machine:

- 62f8604: 3.20 req/s
- efd3441: 4.40 req/s (+37.5%)


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
