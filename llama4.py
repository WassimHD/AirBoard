import requests, os


# API_KEY = os.getenv("OPENROUTER_API_KEY")
# API_KEY = os.getenv("OPENROUTER_API_KEY")

API_KEY="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"


url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
  "Authorization": f"Bearer {API_KEY}",
  "Content-Type": "application/json"
}
payload = {
  "model": "meta-llama/llama-4-scout:free",   # example model name
  "messages": [
    { "role": "user", "content": "do you know a company calle Bed and Sun ?" }
  ],
  "max_tokens": 50
}
resp = requests.post(url, headers=headers, json=payload)
print(resp.json()["choices"][0]["message"]["content"])
