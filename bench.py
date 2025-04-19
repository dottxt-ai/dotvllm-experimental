import asyncio
import time
import json
from typing import List
import statistics
import argparse
from openai import AsyncOpenAI

# Example JSON schemas of varying complexity
SCHEMAS = {
    "simple": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
    },
    "nested": {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                        },
                    },
                },
            }
        },
    },
    "array": {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "string"}, "maxItems": 5}
        },
    },
}


async def make_request(client: AsyncOpenAI, prompt: str, schema: dict) -> float:
    """Make a single request and return completion time"""
    start = time.time()

    await client.completions.create(
        model="gpt2",
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        extra_body={"guided_json": json.dumps(schema)},
    )

    return time.time() - start


async def run_concurrent_requests(
    client: AsyncOpenAI, schema: dict, num_requests: int, prompt: str
) -> List[float]:
    """Run multiple requests concurrently"""
    tasks = [make_request(client, prompt, schema) for _ in range(num_requests)]
    return await asyncio.gather(*tasks)


async def benchmark(base_url: str, concurrency_levels: List[int], prompt: str):
    """Run benchmarks across different schemas and concurrency levels"""
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="not-needed",  # Or whatever is required for your setup
    )

    results = {}

    for schema_name, schema in SCHEMAS.items():
        results[schema_name] = {}
        print(f"\nTesting schema: {schema_name}")

        for num_requests in range(2, concurrency_levels):
            print(f"Running {num_requests} concurrent requests...")

            completion_times = await run_concurrent_requests(
                client, schema, num_requests, prompt
            )

            stats = {
                "mean": statistics.mean(completion_times),
                "median": statistics.median(completion_times),
                "stddev": statistics.stdev(completion_times),
                "throughput": num_requests / max(completion_times),
            }

            results[schema_name][num_requests] = stats

            print(f"Results for {num_requests} requests:")
            print(f"  Mean time: {stats['mean']:.3f}s")
            print(f"  Median time: {stats['median']:.3f}s")
            print(f"  Throughput: {stats['throughput']:.2f} req/s")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--prompt", default="Generate a JSON object with user details:")
    parser.add_argument("--max-concurrency", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(benchmark(args.url, args.max_concurrency, args.prompt))


if __name__ == "__main__":
    main()
