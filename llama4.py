import requests, os, json

#open json file to get api key  
def get_api_key():
    with open('keys.json') as f:
        data = json.load(f)
        return data['llama_4_api_key']



def llama4_scout(query):
    try:
        query=f"Answer the following question shortly and precisely : {query}"
        API_KEY=get_api_key()
        print("Using API Key:", API_KEY)
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/llama-4-scout:free",  
            "messages": [
                { "role": "user", "content": query }
            ],
            "max_tokens": 200
        }

        resp = requests.post(url, headers=headers, json=payload)
        data = resp.json()
        
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        else:
            print("Error response:", data)
            return None

    except Exception as e:  
        print("Error:", e)
        return None