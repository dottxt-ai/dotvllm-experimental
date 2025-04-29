"""DotLLM CompilationManager for non-blocking logits processor compilation."""

import hashlib
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

logger = logging.getLogger("dotllm.compilation_manager")


def make_key(model_name: str, schema: str):
    return hashlib.sha256(f"{model_name}:{schema}".encode("utf-8")).hexdigest()


class CompilationManager:
    """Manager for asynchronous compilation tasks.

    `CompilationManager` provides a process pool for running compilation tasks
    in the background.

    To be able to run compilation in a process pool we need to
    serialize/deserialize the index, which might incur a performance penalty:
    https://github.com/dottxt-ai/dotregex/issues/335.

    A production version would include a better caching mechanism, the cache
    manager would be attached to this class.

    """

    def __init__(self):
        """Initialize the CompilationManager."""
        self.process_pool = ProcessPoolExecutor()
        self._indexes = {}
        self._futures = {}

    def submit(self, func: Callable, model_name: str, schema: str) -> str:
        """Submit a task to be executed in the process pool.

        Args:
            func: The function to execute in the process pool.
            model_name: The name of the model.
            schema: The schema or pattern to compile.
            guide_creator: Function to create a Guide from a deserialized Index.

        Returns:
            A key representing the compilation task.
        """
        key = make_key(model_name, schema)
        if key not in self._futures and key not in self._indexes:
            self._futures[key] = self.process_pool.submit(func, model_name, schema)

        return key

    def get_index(self, key: str):
        """Get the index corresponding to `key`

        Args:
            key: The index's key

        Returns:
            A serialized index
        """
        if key in self._indexes:
            return self._indexes[key]

        try:
            if key not in self._futures:
                logger.error(f"No future found for key {key}")
                return None

            serialized_index = self._futures[key].result()

            self._indexes[key] = serialized_index
            del self._futures[key]
            return serialized_index
        except Exception as e:
            logger.error(f"Guide compilation failed: {e}")
            return None
