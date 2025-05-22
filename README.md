# DotvLLM

DotvLLM is a vLLM wrapper that serves an OpenAI-compatible API using custom logit processors.

⚠️ This is an experimental project. Not for production use. ⚠️

## Architecture

The idea is to be able to replace the guided decoding implementation in vLLM while (1) having a CLI that is a drop-in replacement for `vllm` (2) Spawning the same OpenAI-compatible API as vLLM does. All while adding a minimal amount of code.

To replace the guided decoding implementation we subclass `AsyncLLMEngine` and override `_AsyncLLMEngine.add_request_async`. The subclass instantiates a `CompilationManager` class which uses a `ProcessPoolExecutor` to compile indexes and caches them. The `LogitsProcessor` class is in charge of holding the guide in memory and computing the allowed tokens at each step.

- `api_engine.py`. Most of the code in this module is copied from vLLM, we modified one line to be able to initialize the server with our subclass of `AsyncLLMEngine`.
- `engine.py`. Contains the `AsyncLLMEngine` and `_AsyncLLMEngine` subclasses. We only need a minimal change in `add_request_async` to replace vLLM's guided decoding with our custom implementation.
- `logits_processors.py` dispatches the structure definition to the different backend. Contains the `LogitsProcessor` implementation.
- `dotregex.py`, `dotgrammar.py`, `dotjson.py` contain the code necessary to compile and serialize an index, and build a guide from a serialized index.
- `compilation_manager.py` contains a `CompilationManager` class that uses a `ProcesssPoolExecutor` to compile indexes in parallel, and caches them.


## The sharp bits

- We are forcing vLLM to use the V0 code paths. V1 has a different `LLMEngine` implementation.
- We use a `ProcessPool` instead of a `ThreadPool` to compile the indexes in parallel. As a result we need to serialize/deserialize the indexes, which incurs [a performance penalty](https://github.com/dottxt-ai/dotregex/issues/335).
- The server shuts down whenever the generation fails because of an error with the index. This is on purpose, exceptions that are raised in a task cannot be caught and propagated downstream and returned as an error. We *want* to get the error message as this corresponds to a bug in our structured generation algorithm.
- The server shuts down whenever the compilation fails, because unlike vLLM we add the request even if the index hasn't compiled yet. This can be avoided by checking the validity of the schema before queueing it for compilation.


## Optimizations

Here a few optimization ideas, that we could implement now that we use a subclass of `AsyncLLMEngine`:

- Run the compilation during the KV-cache computation and only schedule the request for generation once compilation is done. This way we're not blocking the generation and reduce the throughput.
- Compute the mask and load it on GPU during the forward pass.


## Installation

```bash
# Install from source
pip install -e .
```

Which will also install vLLM.

## Usage

`dovtllm` wraps vLLM, and it exposes the same OpenAI API. To run a server:

```bash
# Start the API server with all vLLM OpenAI serve options
dotvllm --model <model_id_or_path> --host 0.0.0.0 --port 8000

# Use with OpenAI client
OPENAI_API_BASE=http://localhost:8000 OPENAI_API_KEY=dummy python -c "import openai; print(openai.ChatCompletion.create(model='mistralai/Mistral-7B-Instruct-v0.2', messages=[{'role': 'user', 'content': 'Hello!'}]))"
```

## CLI Options

DotLLM inherits all the CLI options from vLLM's OpenAI API server. You can use the same parameters:

```bash
dotvllm --help
```
