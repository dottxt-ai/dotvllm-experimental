import openai

# Set the API base URL to your DotLLM server
# Default is http://localhost:8000
DOTLLM_API_URL = "http://0.0.0.0:8000/v1"

# API key can be any string (DotLLM doesn't validate it)
client = openai.OpenAI(base_url=DOTLLM_API_URL, api_key="dummy")

# # Send a completion request
response = client.completions.create(
    model="gpt2",  # Use the model name provided to DotLLM
    prompt="What is the capital of France?",
    max_tokens=100,
    temperature=0.7,
)

# # Print the response
print(response.choices[0].text)

# Send a completion request
response = client.completions.create(
    model="gpt2",  # Use the model name provided to DotLLM
    prompt="What is the capital of France?",
    max_tokens=100,
    temperature=0.7,
    extra_body={"guided_regex": "[0-9]{3,8}"},
)

print(response.choices[0].text)


json_grammar = r"""
?value: dict
      | list
      | string
      | SIGNED_NUMBER      -> number
      | "true"             -> true
      | "false"            -> false
      | "null"             -> null

list : "[" [value ("," value)*] "]"

dict : "{" [pair ("," pair)*] "}"
pair : string ":" value

string : ESCAPED_STRING

ESCAPED_STRING: /"(?:[^"\\]|\\.)*"/
// _STRING_INNER: /.*?/
// _STRING_ESC_INNER: _STRING_INNER /(?<!\\)(\\\\)*?/
// ESCAPED_STRING : "\"" _STRING_ESC_INNER "\""
// %import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
"""

response = client.completions.create(
    model="gpt2",  # Use the model name provided to DotLLM
    prompt="What is the capital of France?",
    max_tokens=100,
    temperature=0.7,
    extra_body={"guided_grammar": json_grammar},
)

# Print the response
print(response.choices[0].text)


response = client.completions.create(
    model="gpt2",  # Use the model name provided to DotLLM
    prompt="What is the capital of France?",
    max_tokens=100,
    temperature=0.7,
    extra_body={
        "guided_json": '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}'
    },
)

# Print the response
print(response.choices[0].text)
