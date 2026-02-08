import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "thewindmom/llama3-med42-8b:latest",  # Using the specified model
    "temperature": 0,
    "system": "You are a medical doctor answering real-world exam questions. Answer only with the correct option letter (A, B, C, or D) without any additional explanation.",
    "prompt": """
Question: "In a child with active liver failure, the most important prognosis factor for death is –"

Options:
A) Increasing transaminases
B) Increasing bilirubin
C) Increasing prothrombin time
D) Gram (–)ve sepsis
""",
    "stream": False
}

response = requests.post(url, json=payload)
data = response.json()

if "response" in data:
    print("Response:")
    print(data["response"])
elif "error" in data:
    print("Error:", data["error"])
else:
    print("Unexpected response format:", data)
