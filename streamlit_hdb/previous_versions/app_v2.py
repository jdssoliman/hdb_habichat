from huggingface_hub import InferenceClient
import os

token = os.environ.get("HUGGINGFACE_API_KEY")
client = InferenceClient(token=token)

resp = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Say hello in 5 words."},
    ],
    max_tokens=32,
)
print(resp.choices[0].message.content)
