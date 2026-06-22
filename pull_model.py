import requests

# Direct API call to the local Ollama engine to download Llama3
url = "http://localhost:11434/api/pull"
payload = {"name": "llama3", "stream": False}

print("--- DOWNLOADING LLAMA3 VIA OLLAMA API (This may take a few minutes...) ---")
try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("--- SUCCESS: Llama3 has been downloaded and is ready! ---")
    else:
        print(f"Failed with status code: {response.status_code}, Response: {response.text}")
except Exception as e:
    print(f"Could not connect to Ollama. Is the application running in your system tray? Error: {e}")