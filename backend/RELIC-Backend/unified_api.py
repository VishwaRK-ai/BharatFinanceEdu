import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client

app = FastAPI(title="BharatFinanceEdu Proxy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Render Proxy is running"}

@app.post("/quiz/generate")
async def generate_quiz(request: Request):
    payload = await request.json()
    azure_url = os.getenv("AZURE_API_URL", "http://20.41.114.241:7860").rstrip('/')
    user = os.getenv("AZURE_API_USER", "admin")
    pwd = os.getenv("AZURE_API_PASS", "BharatFinance2026!")
    
    # The official client automatically discovers the correct API endpoint
    client = Client(azure_url, auth=(user, pwd))
    
    # Pass the payload and the required max_tokens (1024)
    result = client.predict(json.dumps(payload), 1024)
    
    try:
        return json.loads(result)
    except Exception:
        return {"response": result}

@app.post("/chat")
async def chat(request: Request):
    payload = await request.json()
    azure_url = os.getenv("AZURE_API_URL", "http://20.41.114.241:7860").rstrip('/')
    user = os.getenv("AZURE_API_USER", "admin")
    pwd = os.getenv("AZURE_API_PASS", "BharatFinance2026!")
    
    client = Client(azure_url, auth=(user, pwd))
    result = client.predict(json.dumps(payload), 1024)
    
    try:
        return json.loads(result)
    except Exception:
        return {"response": result}
