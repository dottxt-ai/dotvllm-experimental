"""DotLLM CompilationManager for non-blocking logits processor compilation."""

import hashlib
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

logger = logging.getLogger("dotllm.compilation_manager")


def get_key(model_name: str, schema: str):
    return hashlib.sha256(f"{model_name}:{schema}".encode("utf-8")).hexdigest()


class CompilationManager:
    """Manager for asynchronous compilation tasks.

    Provides a thread pool for running compilation tasks in the background.
    This simplifies non-blocking compilation of processor indexes.
    """

    def __init__(self, max_workers: int = 4):
        """Initialize the CompilationManager.

        Args:
            max_workers: Maximum number of worker threads in the thread pool.
        """
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._guides = {}
        self._futures = {}

    def submit(
        self, func: Callable, model_name, schema: str
    ) -> concurrent.futures.Future:
        """Submit a task to be executed in the thread pool.

        Args:
            func: The function to execute.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            A Future representing the execution of the task.
        """
        key = get_key(model_name, schema)
        if key not in self._futures and key not in self._guides:
            self._futures[key] = self.thread_pool.submit(func, model_name, schema)

        return key

    def get_guide(self, key: str):
        if key in self._guides:
            return self._guides[key]

        try:
            guide = self._futures[key].result()
            self._guides[key] = guide
            del self._futures[key]
            return guide
        except Exception as e:
            logger.error(f"Guide compilation failed: {e}")
            return None
