import os
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

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
    azure_url = os.getenv("AZURE_API_URL", "").rstrip('/')
    user = os.getenv("AZURE_API_USER", "")
    pwd = os.getenv("AZURE_API_PASS", "")
    
    response = requests.post(f"{azure_url}/api/predict", json={"data": [payload]}, auth=(user, pwd))
    return response.json()

@app.post("/chat")
async def chat(request: Request):
    payload = await request.json()
    azure_url = os.getenv("AZURE_API_URL", "").rstrip('/')
    user = os.getenv("AZURE_API_USER", "")
    pwd = os.getenv("AZURE_API_PASS", "")
    
    response = requests.post(f"{azure_url}/api/predict", json={"data": [payload]}, auth=(user, pwd))
    return response.json()
