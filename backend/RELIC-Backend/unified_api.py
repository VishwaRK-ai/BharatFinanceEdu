import os
import json
import re
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
from gradio_client import Client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Handle Protobuf/LangChain compatibility warnings
os.environ["TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"] = "true"

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

# =====================================================================
# 1. CONFIGURATION & STATE
# =====================================================================

# Render will inject these securely at runtime
raw_api_url = os.getenv("AZURE_API_URL", "http://20.41.114.241:7860/")
match = re.search(r'https?://[^\s\)\]]+', raw_api_url)
api_url = match.group(0) if match else raw_api_url.strip().strip("[]()\"'")
api_user = os.getenv("AZURE_API_USER")
api_pass = os.getenv("AZURE_API_PASS")

llm_client: Optional[Client] = None

def get_client() -> Optional[Client]:
    global llm_client
    if llm_client is not None:
        return llm_client
    try:
        print(f"Connecting to Gradio Client at {api_url}...")
        llm_client = Client(api_url, auth=(api_user, api_pass) if api_user and api_pass else None)
        print("✅ Connected to Gradio LLM successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to Gradio Client. Error: {e}")
        llm_client = None
    return llm_client

# In-memory storage (Replace with Redis/PostgreSQL in production)
chat_history: Dict[str, List[Dict[str, str]]] = {}

vector_db = None


# =====================================================================
# 2. PYDANTIC DATA MODELS
# =====================================================================

class QuizRequest(BaseModel):
    session_id: Optional[str] = None
    sessionId: Optional[str] = None
    target_stock: Optional[str] = None
    targetStock: Optional[str] = None
    ticker: Optional[str] = None
    symbol: Optional[str] = None
    language: Optional[str] = None
    locale: Optional[str] = None
    tip_source: Optional[str] = None
    scan_summary: Optional[Any] = None
    fomo_score: Optional[Any] = None
    fomo_band: Optional[Any] = None

    class Config:
        extra = "allow"

class QuizAnswer(BaseModel):
    question_index: int
    selected_option: str  # "A", "B", "C", or "D"

class QuizSubmit(BaseModel):
    session_id: str
    answers: List[QuizAnswer]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    sessionId: Optional[str] = None
    message: str

    class Config:
        extra = "allow"


# =====================================================================
# 3. CORE HELPER FUNCTIONS (RAG + YFINANCE + LLM BRIDGE)
# =====================================================================

def get_vector_db():
    global vector_db
    if vector_db is not None:
        return vector_db
    try:
        print("Lazy-loading HuggingFaceEmbeddings and ChromaDB...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        print("✅ Local ChromaDB loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load ChromaDB. Error: {e}")
        vector_db = None
    return vector_db

def get_rag_context(query: str, k: int = 3) -> str:
    db = get_vector_db()
    if not db:
        return ""
    try:
        docs = db.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"RAG Error: {e}")
        return ""

def get_live_stock_data(query: str) -> str:
    # A simple entity extractor for the demo (expand with a real NER model later)
    query_lower = query.lower()
    ticker = None
    if "reliance" in query_lower: ticker = "RELIANCE.NS"
    elif "tata motors" in query_lower: ticker = "TATAMOTORS.NS"
    elif "zomato" in query_lower: ticker = "ZOMATO.NS"
    elif "suzlon" in query_lower: ticker = "SUZLON.NS"
    
    if not ticker:
        return ""
        
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get("currentPrice", "Unknown")
        beta = info.get("beta", 1.0)
        return f"[LIVE MARKET DATA for {ticker}]: Current Price: ₹{price}. Risk Profile (Beta): {beta:.2f}."
    except Exception as e:
        print(f"yfinance Error: {e}")
        return ""

def call_colab_llm(user_prompt: str, context: str, system_prompt: str) -> str:
    """Bridges the local FastAPI server to the Colab GPU Inference Server."""
    client = get_client()
    if client is None:
        raise HTTPException(status_code=503, detail="AI Server is currently unreachable.")
    try:
        # Calls the function on remote Azure VM Gradio server
        try:
            response = client.predict(
                prompt=f"{system_prompt}\n\nContext:\n{context}\n\nUser:\n{user_prompt}",
                max_tokens=256,
                api_name="/predict"
            )
        except Exception:
            response = client.predict(
                user_prompt,
                context,
                system_prompt,
                fn_index=0
            )
        return response
    except Exception as e:
        print(f"LLM Bridge Error: {e}")
        raise HTTPException(status_code=503, detail="AI Server is currently unreachable.")

def extract_json_from_llm(text: str) -> List[Dict]:
    """Cleans up the LLM output in case it wraps the JSON in markdown blocks."""
    try:
        match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM failed to output valid JSON.")


# =====================================================================
# 4. FASTAPI APP & ENDPOINTS
# =====================================================================

app = FastAPI(title="BharatFinanceEdu API", version="1.0")

@app.get("/")
def health_check():
    return {"status": "Active", "message": "RelicLLM API is running locally."}


# --- QUIZ ENDPOINTS ---

@app.get("/quiz")
def get_quiz_info():
    return {
        "message": "Welcome to the Adaptive Suitability Check. Request a stock to begin.",
        "supported_stocks": ["Reliance", "Tata Motors", "Zomato", "Suzlon"]
    }

@app.post("/quiz")
@app.post("/quiz/generate")
def generate_quiz(req: QuizRequest):
    stock_ticker = req.target_stock or req.targetStock or req.ticker or req.symbol or "RELIANCE.NS"
    session_id = req.session_id or req.sessionId or "session"

    try:
        # 1. Fetch live beta and context for the target stock
        live_data = get_live_stock_data(stock_ticker)
        rag_context = get_rag_context(f"risks of investing in {stock_ticker}")
        combined_context = f"{live_data}\n\nSEBI Guidelines:\n{rag_context}"
        
        # 2. Force the LLM to act as a strict JSON generator
        system_prompt = (
            "You are a strict JSON Quiz Generator for an Indian financial education app. "
            "Generate exactly 3 multiple-choice questions assessing the user's readiness to invest in the requested stock. "
            "Question 1 must be about Risk Tolerance (use the provided Beta metric). "
            "Question 2 must be about the Business Model. "
            "Question 3 must be about Capital Allocation/Time Horizon. "
            "Output ONLY a valid JSON array of objects with keys: 'question', 'options' (A, B, C, D dict), and 'correct_answer' (A, B, C, or D). "
            "Do not include markdown ticks."
        )
        
        raw_response = call_colab_llm(stock_ticker, combined_context, system_prompt)
        quiz_json = extract_json_from_llm(raw_response)
    except Exception as e:
        print(f"Quiz generation LLM error (using fallback): {e}")
        quiz_json = [
            {
                "category": "Risk Tolerance",
                "question": f"Direct investments in equities like {stock_ticker} carry volatility. How should you approach sudden market pullbacks?",
                "options": {
                    "A": "Panic and sell out entirely at the first dip.",
                    "B": "Hold or accumulate if long-term business fundamentals remain solid.",
                    "C": "Borrow money on high leverage to speculate.",
                    "D": "Stop tracking your finances completely."
                },
                "correct_answer": "B",
                "explanation": "Equity volatility is normal; disciplined asset allocation prevents panic selling."
            },
            {
                "category": "Business Model",
                "question": f"Before investing in {stock_ticker}, what research is essential?",
                "options": {
                    "A": "Relying on anonymous tips from viral messaging channels.",
                    "B": "Reviewing quarterly profit trends, operating cash flows, and debt levels.",
                    "C": "Checking only if the price rose sharply yesterday.",
                    "D": "Acting purely on social media hype."
                },
                "correct_answer": "B",
                "explanation": "SEBI encourages retail investors to base decisions on official filings, not unverified tips."
            },
            {
                "category": "Capital Allocation",
                "question": f"What portion of your total portfolio is prudent to allocate to a single stock like {stock_ticker}?",
                "options": {
                    "A": "More than 50% in a concentrated bet.",
                    "B": "Under 10-15% as part of a well-diversified basket.",
                    "C": "100% using emergency funds.",
                    "D": "Money needed for bills within the next 30 days."
                },
                "correct_answer": "B",
                "explanation": "Diversification limits idiosyncratic business risks across your wealth."
            }
        ]
    
    # Return complete quiz data directly to the Next.js frontend (stateless)
    return {"session_id": session_id, "ticker": stock_ticker, "quiz": quiz_json}

@app.post("/quiz/submit")
def submit_quiz(req: QuizSubmit):
    # Deprecated: quiz grading is now stateless and performed by the frontend against Neon DB.
    return {
        "detail": "Quiz grading is stateless. Evaluation is performed by the frontend application against persistent database records."
    }


# --- ASSISTANT ENDPOINTS ---

@app.get("/assistant/history/{session_id}")
def get_chat_history(session_id: str):
    return {"session_id": session_id, "history": chat_history.get(session_id, [])}

@app.post("/assistant")
@app.post("/chat")
def chat_with_assistant(req: ChatRequest):
    session_id = req.session_id or req.sessionId or "session"
    # 1. RAG and Live Data Synthesis
    rag_context = get_rag_context(req.message)
    live_data = get_live_stock_data(req.message)
    combined_context = f"{rag_context}\n\n{live_data}" if live_data else rag_context

    # 2. Retrieve history and format for prompt context
    history = chat_history.get(session_id, [])
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]]) # Keep last 4 messages

    full_user_prompt = f"Previous Conversation:\n{history_text}\n\nCurrent User Query: {req.message}" if history else req.message

    # 3. Setup standard assistant persona
    system_prompt = (
        "You are BharatFinanceEdu, an expert Indian financial educator. "
        "Explain concepts simply, using analogies and dispelling misconceptions. "
        "Base your answers on the provided context. If the user acts impulsively, warn them based on SEBI rules."
    )

    # 4. Generate Response
    ai_response = call_colab_llm(full_user_prompt, combined_context, system_prompt)

    # 5. Update session memory
    if session_id not in chat_history:
        chat_history[session_id] = []
    chat_history[session_id].append({"role": "User", "content": req.message})
    chat_history[session_id].append({"role": "AI", "content": ai_response})

    return {
        "reply": ai_response,
        "rag_sources_used": bool(rag_context),
        "live_data_used": bool(live_data)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)