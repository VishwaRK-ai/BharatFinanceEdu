import os
import re
import json
import logging
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client

logger = logging.getLogger("unified_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="BharatFinanceEdu Proxy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STOCK_NAMES = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "TATAMOTORS.NS": "Tata Motors",
    "ITC.NS": "ITC Limited",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "BAJFINANCE.NS": "Bajaj Finance",
    "WIPRO.NS": "Wipro",
    "ZOMATO.NS": "Zomato",
    "SUZLON.NS": "Suzlon Energy",
}


def get_stock_display_name(ticker: str) -> str:
    clean = ticker.upper().strip()
    if clean in STOCK_NAMES:
        return STOCK_NAMES[clean]
    base = clean.replace(".NS", "").replace(".BO", "")
    return base.title()


def generate_fallback_quiz(target_stock: str, session_id: str = "session_1") -> Dict[str, Any]:
    """Deterministic, high-quality stock-specific fallback quiz conforming to normalizeQuizStart schema."""
    stock_name = get_stock_display_name(target_stock)
    questions = [
        {
            "category": "Risk Tolerance",
            "question": f"Before investing in {stock_name} ({target_stock}), what should you check regarding downside risk and volatility?",
            "options": {
                "A": "Ignore price fluctuations because equity prices always recover immediately.",
                "B": "Review its historical Beta and ensure you are comfortable with potential 15-20% corrections.",
                "C": "Borrow high-interest personal loans to double your position during a steep fall.",
                "D": "Rely entirely on anonymous social media advice without checking risk factors."
            },
            "correct_answer": "B",
            "explanation": "SEBI emphasizes understanding Beta and drawdown tolerance before allocating capital to equities."
        },
        {
            "category": "Valuation",
            "question": f"How should an investor evaluate the current valuation of {stock_name} ({target_stock})?",
            "options": {
                "A": "Buy immediately only because the stock hit a new 52-week high yesterday.",
                "B": "Assume a high P/E ratio guarantees high future returns.",
                "C": "Compare its P/E, P/B, and earnings growth relative to sector peers and historical medians.",
                "D": "Avoid looking at financial reports and rely exclusively on rumors."
            },
            "correct_answer": "C",
            "explanation": "Evaluating multiples relative to historical medians and sector peers provides a margin of safety."
        },
        {
            "category": "Business Model",
            "question": f"When researching {stock_name}'s long-term business sustainability, which metric is most reliable?",
            "options": {
                "A": "Consistent operating cash flows, revenue growth, and quarterly profit trends.",
                "B": "Short-term momentum on intraday chat forums.",
                "C": "The popularity of the company's brand among friends.",
                "D": "Sensational headlines promising multi-bagger gains."
            },
            "correct_answer": "A",
            "explanation": "Sustainable shareholder value is driven by healthy operating cash flows and consistent earnings."
        },
        {
            "category": "Capital Allocation",
            "question": f"According to SEBI investor guidelines, what is the safest way to size your investment in {stock_name}?",
            "options": {
                "A": "Invest 100% of your savings into this single company for maximum upside.",
                "B": "Limit exposure to under 10-15% of your equity portfolio as part of a diversified basket.",
                "C": "Use emergency reserves meant for health and household expenses.",
                "D": "Take leveraged futures positions without any stop-loss."
            },
            "correct_answer": "B",
            "explanation": "Disciplined portfolio diversification limits the damage if an individual stock faces adverse events."
        }
    ]
    return {
        "session_id": session_id,
        "quiz_id": session_id,
        "ticker": target_stock,
        "stock_name": stock_name,
        "topic": "Investor Readiness & Risk Management",
        "quiz": questions,
        "questions": questions
    }


def extract_and_validate_quiz(
    raw_payload: Any,
    target_stock: str,
    session_id: str
) -> Optional[Dict[str, Any]]:
    """Extracts, cleans, and validates structured quiz JSON conforming to frontend normalizeQuizStart."""
    parsed = None
    if isinstance(raw_payload, dict):
        parsed = raw_payload
    elif isinstance(raw_payload, str):
        cleaned = raw_payload.strip()
        # Strip markdown fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
        except Exception:
            match_obj = re.search(r"\{[\s\S]*\}", cleaned)
            if match_obj:
                try:
                    parsed = json.loads(match_obj.group(0))
                except Exception:
                    pass
            if parsed is None:
                match_arr = re.search(r"\[[\s\S]*\]", cleaned)
                if match_arr:
                    try:
                        parsed = {"quiz": json.loads(match_arr.group(0))}
                    except Exception:
                        pass

    if not isinstance(parsed, dict) and not isinstance(parsed, list):
        return None

    if isinstance(parsed, list):
        parsed = {"quiz": parsed}

    raw_questions = None
    if isinstance(parsed.get("quiz"), list) and len(parsed["quiz"]) > 0:
        raw_questions = parsed["quiz"]
    elif isinstance(parsed.get("questions"), list) and len(parsed["questions"]) > 0:
        raw_questions = parsed["questions"]
    elif isinstance(parsed.get("data"), dict) and isinstance(parsed["data"].get("questions"), list):
        raw_questions = parsed["data"]["questions"]

    if not raw_questions or not isinstance(raw_questions, list):
        return None

    validated_questions = []
    default_categories = ["Risk Tolerance", "Valuation", "Business Model", "Capital Allocation"]

    for idx, item in enumerate(raw_questions):
        if not isinstance(item, dict):
            continue

        q_text = (
            item.get("question")
            or item.get("question_text")
            or item.get("text")
            or item.get("prompt")
        )
        if not q_text or not isinstance(q_text, str) or not q_text.strip():
            continue
        q_text = q_text.strip()

        opts_raw = item.get("options") or item.get("choices")
        norm_options = {}
        if isinstance(opts_raw, dict):
            for k, v in opts_raw.items():
                k_clean = str(k).strip().upper()
                if k_clean in ("A", "B", "C", "D"):
                    norm_options[k_clean] = str(v).strip()
        elif isinstance(opts_raw, list) and len(opts_raw) >= 4:
            for l_idx, letter in enumerate(["A", "B", "C", "D"]):
                norm_options[letter] = str(opts_raw[l_idx]).strip()

        if not all(norm_options.get(l) for l in ["A", "B", "C", "D"]):
            continue

        raw_ans = (
            item.get("correct_answer")
            or item.get("correctAnswer")
            or item.get("answer")
            or "A"
        )
        match_ans = re.search(r"([A-D])", str(raw_ans).upper())
        correct_ans = match_ans.group(1) if match_ans else "A"

        explanation = str(
            item.get("explanation")
            or item.get("reason")
            or "Prudent, SEBI-aligned financial evaluation choice."
        ).strip()

        category = str(
            item.get("category")
            or item.get("topic")
            or default_categories[idx % len(default_categories)]
        ).strip()

        validated_questions.append({
            "category": category,
            "question": q_text,
            "options": norm_options,
            "correct_answer": correct_ans,
            "explanation": explanation
        })

    if len(validated_questions) < 2:
        return None

    stock_name = (
        parsed.get("stock_name")
        or parsed.get("stockName")
        or get_stock_display_name(target_stock)
    )
    topic = parsed.get("topic") or "Investor Readiness & Risk Management"

    return {
        "session_id": session_id,
        "quiz_id": session_id,
        "ticker": target_stock,
        "stock_name": stock_name,
        "topic": topic,
        "quiz": validated_questions,
        "questions": validated_questions
    }


@app.get("/")
def read_root():
    return {"status": "Render Proxy is running"}


@app.post("/language")
async def set_language(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    return {
        "session_id": data.get("session_id", "default"),
        "language": data.get("language", "en")
    }


@app.post("/quiz/generate")
@app.post("/quiz")
async def generate_quiz(request: Request):
    payload = await request.json()
    target_stock = (
        payload.get("target_stock")
        or payload.get("targetStock")
        or payload.get("ticker")
        or "RELIANCE.NS"
    )
    session_id = str(payload.get("session_id") or payload.get("sessionId") or "default_session")

    # Explicitly flag action as quiz for Azure LLM handler
    payload["action"] = "quiz"

    azure_url = os.getenv("AZURE_API_URL", "http://20.41.114.241:7860").rstrip('/')
    user = os.getenv("AZURE_API_USER", "admin")
    pwd = os.getenv("AZURE_API_PASS", "BharatFinance2026!")

    result = None
    try:
        client = Client(azure_url, auth=(user, pwd), httpx_kwargs={"timeout": 120.0})
        result = client.predict(json.dumps(payload), 1024)
        logger.info(f"Received Azure prediction response (length {len(str(result))})")
    except Exception as e:
        print(f"Error: [Quiz Azure Prediction Failed] {e}")
        logger.error(f"[Render->Azure Quiz Error] Connection/prediction to {azure_url} failed: {e}", exc_info=True)

    # Validate and normalize result into schema expected by normalizeQuizStart
    validated = extract_and_validate_quiz(result, target_stock, session_id)
    if validated:
        logger.info(f"Successfully normalized quiz for {target_stock} with {len(validated['questions'])} questions.")
        return validated

    logger.warning(f"Could not parse valid quiz from Azure response. Serving deterministic fallback quiz for {target_stock}.")
    return generate_fallback_quiz(target_stock, session_id)


@app.post("/chat")
@app.post("/assistant")
async def chat(request: Request):
    payload = await request.json()
    azure_url = os.getenv("AZURE_API_URL", "http://20.41.114.241:7860").rstrip('/')
    user = os.getenv("AZURE_API_USER", "admin")
    pwd = os.getenv("AZURE_API_PASS", "BharatFinance2026!")

    try:
        client = Client(azure_url, auth=(user, pwd), httpx_kwargs={"timeout": 120.0})
        result = client.predict(json.dumps(payload), 1024)
    except Exception as e:
        print(f"Error: [Chat Azure Prediction Failed] {e}")
        logger.error(f"[Render->Azure Chat Error] Connection/prediction to {azure_url} failed: {e}", exc_info=True)
        return {
            "reply": "I am currently connecting to market intelligence servers. Please ask your question again in a moment.",
            "rag_sources_used": False,
            "live_data_used": False
        }

    # Clean and format response for frontend assistantResponseSchema
    reply_text = str(result).strip()
    if reply_text.startswith("{") and reply_text.endswith("}"):
        try:
            data = json.loads(reply_text)
            if "reply" in data:
                return data
            if "response" in data:
                reply_text = str(data["response"]).strip()
        except Exception:
            pass

    return {
        "reply": reply_text,
        "rag_sources_used": True,
        "live_data_used": False
    }
