import asyncio
import base64
import io
import json
import logging
import os
import random
import re
import secrets
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from gradio_client import Client
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError, model_validator

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ["TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"] = "true"

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from asr_and_tts import run_multilingual_asr, synthesize_speech
from fomo_scorer import evaluate_fomo
from live_stock_api import extract_ticker, get_live_stock_data
from test_ocr_security import analyze_image
from volatility_alert import check_volatility

# Render will inject these securely at runtime
raw_api_url = os.getenv("AZURE_API_URL", "http://20.41.114.241:7860/")
# Strip potential markdown link formatting e.g. [url](url) or surrounding brackets/quotes/whitespace
match = re.search(r'https?://[^\s\)\]]+', raw_api_url)
api_url = match.group(0) if match else raw_api_url.strip().strip("[]()\"'")
api_user = os.getenv("AZURE_API_USER")
api_pass = os.getenv("AZURE_API_PASS")

COLAB_API_URL = api_url
COLAB_AUTH = (api_user, api_pass) if api_user and api_pass else None
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HISTORY_WINDOW = 6
HISTORY_LIMIT = 40
QUIZ_MIN_QUESTIONS = 5
QUIZ_MAX_QUESTIONS = 6
QUIZ_LLM_ATTEMPTS = 2
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_AUDIO_BYTES = 10 * 1024 * 1024
WATCHLIST = ["RELIANCE.NS", "TATAMOTORS.NS", "ZOMATO.NS", "SUZLON.NS"]
DROP_THRESHOLD = -2.0
POLL_SECONDS = 60
SIMULATE_STREAM = True
DEFAULT_LANGUAGE = "en"
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}

QUIZ_CATEGORIES = [
    "Risk Tolerance",
    "Business Model",
    "Capital Allocation",
    "Valuation",
    "Diversification",
    "Time Horizon",
]

ASSISTANT_PROMPT = (
    "You are BharatFinanceEdu, an expert Indian financial educator. "
    "Base your answers on the provided context. "
    "If the user acts impulsively, warn them based on SEBI rules."
)
PLAIN_MODE = (
    "Plain answer mode: give only a short, simple explanation in 2 to 4 sentences. "
    "Do not add an analogy, examples or misconceptions, and do not use headings, bold text or section labels."
)
DETAILED_MODE = (
    "Detailed mode: the user wants a detailed explanation of the topic being discussed. "
    "Fill in all four sections of this exact template, with two or three sentences in each, "
    "and keep the section labels in English:\n"
    "**Explanation:** ...\n**Analogy:** ...\n**Example:** ...\n**Common Misconception:** ..."
)
DETAIL_REQUEST = re.compile(
    r"\b(?:in detail|more detail|detailed explanation|in depth|elaborate|explain more|tell me more)\b"
    r"|विस्तार|विस्तृत|सविस्तर|और बताओ|अधिक सांगा",
    re.IGNORECASE,
)
SECTION_LABEL = re.compile(
    r"\*{0,2}[ \t]*(?:Explanation|Analogy|Common Misconceptions?|Misconceptions?|Example)[ \t]*\*{0,2}[ \t]*:[ \t]*\*{0,2}[ \t]*",
    re.IGNORECASE,
)
BOLD_LABEL = re.compile(r"\*\*[^*\n]{1,40}?:\*\*[ \t]*|\*\*[^*\n]{1,40}?\*\*[ \t]*:[ \t]*")
QUIZ_PROMPT = (
    "You are a strict JSON quiz generator for an Indian financial education app. "
    f"Generate {QUIZ_MIN_QUESTIONS} to {QUIZ_MAX_QUESTIONS} multiple-choice questions that test whether the user "
    "understands the current state of the specific stock they are about to buy. "
    f"Cover these areas: {', '.join(QUIZ_CATEGORIES)}. "
    "Every question must quote at least one figure or fact from the LIVE SNAPSHOT, such as the recent returns, "
    "distance from the 52-week high, Beta, P/E, P/B, quarterly results trend, debt, cash flow, promoter or "
    "institutional holding, recent news or the next results date. "
    "Pitch the questions at an intermediate retail investor: do not ask trivia such as founding year, "
    "CEO names or exact numbers to memorise, and do not ask generic definitions that ignore this stock. "
    "Ask what the figures mean for a buyer today. "
    "Risk Tolerance: Beta, expected drawdown, recent price swings, debt level and recent news. "
    "Valuation: P/E, P/B and where the price sits in its 52-week range. "
    "Business Model: the business segments, quarterly revenue and profit trend, margins and cash flow. "
    "Capital Allocation: how much of a portfolio suits a stock with this risk and holding pattern. "
    "Diversification: concentration in this sector and reliance on a single company. "
    "Time Horizon: holding period suited to this volatility, momentum and the upcoming results date. "
    "The correct answer must be the most prudent, SEBI-aligned choice and the wrong options must be plausible. "
    "Output ONLY a valid JSON array where each object has keys 'category' (one of the areas above), "
    "'question', 'options' (an object with keys A, B, C, D), 'correct_answer' (one of A, B, C, D) "
    "and 'explanation' (one sentence on why the correct answer is right). "
    "Keep each question under 30 words and each explanation under 25 words. "
    "Do not repeat the options inside the question text. "
    "Do not include markdown fences or any other text."
)

logger = logging.getLogger("relic")

llm_client: Optional[Client] = None
vector_db: Optional[Any] = None
chat_history: Dict[str, List[Dict[str, str]]] = {}
session_languages: Dict[str, str] = {}
market_state: Dict[str, Any] = {"checked_at": None, "alerts": {}}


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    sessionId: Optional[str] = None
    message: str

    class Config:
        extra = "allow"


class ChatResponse(BaseModel):
    reply: str
    detailed: bool
    rag_sources_used: bool
    live_data_used: bool


class LanguageRequest(BaseModel):
    session_id: str
    language: str


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
    selected_option: Literal["A", "B", "C", "D"]


class QuizSubmit(BaseModel):
    session_id: str
    answers: List[QuizAnswer]


class QuizQuestion(BaseModel):
    category: str
    question: str
    options: Dict[Literal["A", "B", "C", "D"], str]
    correct_answer: Literal["A", "B", "C", "D"]
    explanation: str

    @model_validator(mode="after")
    def validate_options(self) -> "QuizQuestion":
        if len(self.options) != 4:
            raise ValueError("each question needs exactly 4 options")
        return self


class SpeakRequest(BaseModel):
    session_id: str
    text: str


class FomoRequest(BaseModel):
    message: str
    ticker: Optional[str] = None
    demo_mode: bool = False


def get_vector_db() -> Optional[Any]:
    global vector_db
    if vector_db is not None:
        return vector_db
    try:
        logger.info("Lazy-loading HuggingFaceEmbeddings and ChromaDB...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        logger.info("ChromaDB loaded successfully.")
    except Exception:
        logger.exception("ChromaDB unavailable, continuing without RAG")
        vector_db = None
    return vector_db


def get_rag_context(query: str, k: int = 3) -> str:
    db = get_vector_db()
    if db is None:
        return ""
    try:
        docs = db.similarity_search(query, k=k)
        return "\n".join(doc.page_content for doc in docs)
    except Exception:
        logger.exception("RAG lookup failed")
        return ""


def call_colab_llm(prompt: str, context: str, system_prompt: str) -> str:
    global llm_client
    try:
        if llm_client is None:
            llm_client = Client(COLAB_API_URL, auth=COLAB_AUTH)
        try:
            return str(llm_client.predict(prompt=f"{system_prompt}\n\nContext:\n{context}\n\nUser:\n{prompt}", max_tokens=256, api_name="/predict"))
        except Exception:
            return str(llm_client.predict(prompt, context, system_prompt, fn_index=0))
    except Exception:
        llm_client = None
        logger.exception("Colab inference call failed")
        raise HTTPException(status_code=503, detail="AI server is currently unreachable.")


def normalize_question(item: Dict[str, Any]) -> QuizQuestion:
    return QuizQuestion(
        category=str(item.get("category") or "General").strip().title(),
        question=re.split(r"\n\s*A[).:]", str(item["question"]))[0].strip(),
        options={str(k).strip().upper()[:1]: str(v).strip() for k, v in item["options"].items()},
        correct_answer=str(item["correct_answer"]).strip().upper()[:1],
        explanation=str(item.get("explanation") or "").strip(),
    )


def shuffle_options(question: QuizQuestion) -> QuizQuestion:
    letters = sorted(question.options)
    texts = [question.options[letter] for letter in letters]
    correct_text = question.options[question.correct_answer]
    random.shuffle(texts)
    return question.model_copy(
        update={"options": dict(zip(letters, texts)), "correct_answer": letters[texts.index(correct_text)]}
    )


def parse_quiz(text: str) -> List[QuizQuestion]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    questions = [shuffle_options(normalize_question(item)) for item in json.loads(match.group(0) if match else cleaned)]
    if not QUIZ_MIN_QUESTIONS <= len(questions) <= QUIZ_MAX_QUESTIONS:
        raise ValueError(f"expected {QUIZ_MIN_QUESTIONS}-{QUIZ_MAX_QUESTIONS} questions, got {len(questions)}")
    return questions


LEVEL_ADVICE = {
    "Well Prepared": "Your understanding matches the risks of this stock; proceed only with disciplined position sizing.",
    "Partially Prepared": "Revise the weak areas below first, and if you still invest, start with a small position.",
    "Not Yet Ready": "Avoid buying for now. SEBI advises against investing in what you do not understand; "
    "learn the points below first.",
}


def readiness_level(percentage: float) -> str:
    if percentage >= 80:
        return "Well Prepared"
    if percentage >= 50:
        return "Partially Prepared"
    return "Not Yet Ready"


def speak(text: str, language: str) -> bytes:
    clean = " ".join(re.sub(r"[*#_`>~|]+", " ", text).split())
    if not clean:
        raise HTTPException(status_code=400, detail="Nothing to speak after removing formatting.")
    try:
        return synthesize_speech(clean, language)
    except Exception:
        logger.exception("TTS failed for language %s", language)
        raise HTTPException(status_code=503, detail="Speech service is currently unreachable.")


def transcribe(upload: UploadFile) -> str:
    data = upload.file.read(MAX_AUDIO_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio exceeds 10 MB limit.")
    suffix = os.path.splitext(upload.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
    try:
        text, _ = run_multilingual_asr(tmp.name)
    except Exception:
        logger.exception("ASR failed")
        raise HTTPException(status_code=422, detail="Could not transcribe the audio.")
    finally:
        os.remove(tmp.name)
    if not text.strip():
        raise HTTPException(status_code=422, detail="No speech detected.")
    return text.strip()


def strip_section_labels(text: str) -> str:
    text = SECTION_LABEL.sub("", text)
    text = BOLD_LABEL.sub("", text)
    return text.replace("**", "").strip()


def run_chat(session_id: str, message: str) -> ChatResponse:
    rag_context = get_rag_context(message)
    ticker = extract_ticker(message)
    live_data = get_live_stock_data(ticker) if ticker else ""
    context = "\n\n".join(part for part in (rag_context, live_data) if part)

    language = LANGUAGES[session_languages.get(session_id, DEFAULT_LANGUAGE)]
    history = chat_history.setdefault(session_id, [])
    recent = "\n".join(f"{m['role']}: {m['content']}" for m in history[-HISTORY_WINDOW:])
    detailed = bool(DETAIL_REQUEST.search(message))
    style = DETAILED_MODE if detailed else PLAIN_MODE
    query = f"{message}\n\n{style} Answer in {language}."
    prompt = f"Previous Conversation:\n{recent}\n\nCurrent User Query: {query}" if recent else query

    reply = call_colab_llm(prompt, context, f"{ASSISTANT_PROMPT} {style} Always answer in {language}.")
    if not detailed:
        reply = strip_section_labels(reply)

    history.append({"role": "User", "content": message})
    history.append({"role": "AI", "content": reply})
    del history[:-HISTORY_LIMIT]

    return ChatResponse(
        reply=reply, detailed=detailed, rag_sources_used=bool(rag_context), live_data_used=bool(live_data)
    )


def normalize_ticker(target: str) -> str:
    ticker = extract_ticker(target) or target.strip().upper().replace(" ", "")
    return ticker if ticker.endswith((".NS", ".BO")) else f"{ticker}.NS"


def is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value


def pct(value: Any, scale: float = 100) -> str:
    return f"{value * scale:.1f}%" if is_num(value) else "N/A"


def num(value: Any) -> str:
    return f"{value:.2f}" if is_num(value) else "N/A"


def crore(value: Any) -> str:
    return f"Rs {value / 1e7:,.0f} crore" if is_num(value) else "N/A"


def safe(fetch: Callable[[], Any]) -> Any:
    try:
        return fetch()
    except Exception:
        return None


def statement_row(frame: Any, name: str, count: int = 1) -> List[float]:
    if frame is None or getattr(frame, "empty", True) or name not in frame.index:
        return []
    return [float(v) for v in frame.loc[name].dropna().iloc[:count]][::-1]


def holder_pct(frame: Any, key: str) -> Optional[float]:
    value = safe(lambda: float(frame.loc[key].iloc[0]))
    return value if is_num(value) else None


def headlines(items: Any, count: int = 3) -> List[str]:
    titles = []
    for item in items or []:
        title = (item.get("content") or {}).get("title") or item.get("title")
        if title:
            titles.append(title.strip())
    return titles[:count]


def next_results_date(calendar: Any) -> Optional[str]:
    dates = calendar.get("Earnings Date") if isinstance(calendar, dict) else None
    return str(dates[0]) if dates else None


def fetch_stock_snapshot(ticker: str) -> Dict[str, Any]:
    stock = yf.Ticker(ticker)
    info = safe(lambda: stock.info) or {}
    close = safe(lambda: stock.history(period="1y")["Close"])
    if close is None or close.empty:
        price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 1000.0)
        high_52 = float(info.get("fiftyTwoWeekHigh") or price * 1.25)
        low_52 = float(info.get("fiftyTwoWeekLow") or price * 0.75)
        beta = float(info.get("beta") or 1.0)
        ret_1y = 0.05
    else:
        price = float(close.iloc[-1])
        high_52 = info.get("fiftyTwoWeekHigh") or float(close.max())
        low_52 = info.get("fiftyTwoWeekLow") or float(close.min())
        beta = info.get("beta") if is_num(info.get("beta")) else 1.0
        ret_1y = price / float(close.iloc[0]) - 1

    def ret(days: int) -> Optional[float]:
        return price / float(close.iloc[-days]) - 1 if len(close) > days else None

    quarterly = safe(lambda: stock.quarterly_income_stmt)
    balance = safe(lambda: stock.balance_sheet)
    cashflow = safe(lambda: stock.cashflow)
    holders = safe(lambda: stock.major_holders)
    debt = (statement_row(balance, "Total Debt") or [None])[0]
    equity = (statement_row(balance, "Stockholders Equity") or [None])[0]

    return {
        "ticker": ticker,
        "name": info.get("shortName") or ticker.split(".")[0],
        "price": price,
        "beta": round(beta, 2),
        "sector": info.get("sector") or "N/A",
        "industry": info.get("industry") or "N/A",
        "market_cap": info.get("marketCap"),
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "profit_margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "high_52": high_52,
        "low_52": low_52,
        "from_high": price / high_52 - 1,
        "ret_1m": ret(21),
        "ret_6m": ret(126),
        "ret_1y": price / float(close.iloc[0]) - 1,
        "quarterly_revenue": statement_row(quarterly, "Total Revenue", 4),
        "quarterly_profit": statement_row(quarterly, "Net Income", 4),
        "total_debt": debt,
        "cash": (statement_row(balance, "Cash And Cash Equivalents") or [None])[0],
        "debt_to_equity": debt / equity if is_num(debt) and is_num(equity) and equity else None,
        "free_cash_flow": (statement_row(cashflow, "Free Cash Flow") or [None])[0],
        "insiders": holder_pct(holders, "insidersPercentHeld"),
        "institutions": holder_pct(holders, "institutionsPercentHeld"),
        "news": headlines(safe(lambda: stock.news)),
        "next_results": next_results_date(safe(lambda: stock.calendar)),
        "summary": (info.get("longBusinessSummary") or "")[:600],
    }


def build_quiz_context(snap: Dict[str, Any]) -> str:
    lines = [
        f"LIVE SNAPSHOT for {snap['name']} ({snap['ticker']}):",
        f"Sector: {snap['sector']} / {snap['industry']}. Market cap: {crore(snap['market_cap'])}.",
        f"Price: Rs {snap['price']:.2f}. 52-week range: Rs {snap['low_52']:.2f} to Rs {snap['high_52']:.2f} "
        f"({pct(snap['from_high'])} from the 52-week high).",
        f"Returns: 1 month {pct(snap['ret_1m'])}, 6 months {pct(snap['ret_6m'])}, 1 year {pct(snap['ret_1y'])}.",
        f"Beta: {snap['beta']:.2f} (if the NIFTY falls 10%, this stock could fall about {snap['beta'] * 10:.1f}%).",
        f"P/E: {num(snap['pe'])}. P/B: {num(snap['pb'])}. Profit margin: {pct(snap['profit_margin'])}. "
        f"Revenue growth (YoY): {pct(snap['revenue_growth'])}.",
    ]
    if snap["quarterly_revenue"]:
        lines.append("Quarterly revenue, oldest to latest: " + " -> ".join(map(crore, snap["quarterly_revenue"])) + ".")
    if snap["quarterly_profit"]:
        lines.append("Quarterly net profit, oldest to latest: " + " -> ".join(map(crore, snap["quarterly_profit"])) + ".")
    if is_num(snap["total_debt"]):
        lines.append(
            f"Total debt: {crore(snap['total_debt'])}. Cash: {crore(snap['cash'])}. "
            f"Debt to equity: {num(snap['debt_to_equity'])}."
        )
    if is_num(snap["free_cash_flow"]):
        lines.append(f"Free cash flow (last year): {crore(snap['free_cash_flow'])}.")
    if is_num(snap["insiders"]) or is_num(snap["institutions"]):
        lines.append(
            f"Holding: promoters/insiders {pct(snap['insiders'])}, institutions {pct(snap['institutions'])}."
        )
    if snap["next_results"]:
        lines.append(f"Next results date: {snap['next_results']}.")
    if snap["news"]:
        lines.append("Recent news: " + " | ".join(snap["news"]))
    lines.append(f"Business summary: {snap['summary'] or 'N/A'}")
    sebi = get_rag_context(f"risks of investing in {snap['name']}")
    return "\n".join(lines) + f"\n\nSEBI guidance:\n{sebi}"


def build_verdict(name: str, level: str, score: int, total: int, strengths: List[str], gaps: List[str]) -> str:
    parts = [f"{level} to invest in {name}: you answered {score} of {total} correctly."]
    if strengths:
        parts.append(f"You show sound understanding of {', '.join(strengths)}.")
    if gaps:
        parts.append(f"You need to work on {', '.join(gaps)} before committing money.")
    parts.append(LEVEL_ADVICE[level])
    return " ".join(parts)


async def monitor_market() -> None:
    while True:
        for ticker in WATCHLIST:
            try:
                result = await asyncio.to_thread(check_volatility, ticker, DROP_THRESHOLD, SIMULATE_STREAM)
                market_state["alerts"][ticker] = result
            except Exception:
                logger.exception("Volatility check failed for %s", ticker)
        market_state["checked_at"] = datetime.now(timezone.utc).isoformat()
        await asyncio.sleep(POLL_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    monitor = asyncio.create_task(monitor_market())
    yield
    monitor.cancel()


app = FastAPI(title="BharatFinanceEdu API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check() -> Dict[str, Any]:
    return {"status": "active", "rag_loaded": vector_db is not None, "colab_connected": llm_client is not None}


@app.post("/chat", response_model=ChatResponse)
@app.post("/assistant", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or req.sessionId or secrets.token_hex(8)
    return run_chat(session_id, req.message)


@app.get("/chat/history/{session_id}")
def get_chat_history(session_id: str) -> Dict[str, Any]:
    return {"session_id": session_id, "history": chat_history.get(session_id, [])}


@app.get("/languages")
def list_languages() -> Dict[str, Any]:
    return {"default": DEFAULT_LANGUAGE, "languages": LANGUAGES}


@app.post("/language")
def set_language(req: LanguageRequest) -> Dict[str, str]:
    if req.language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language. Choose from {list(LANGUAGES)}.")
    session_languages[req.session_id] = req.language
    return {"session_id": req.session_id, "language": req.language}


def generate_fallback_quiz(snap: Dict[str, Any]) -> List[QuizQuestion]:
    name = snap.get("name") or snap.get("ticker", "Stock")
    beta = float(snap.get("beta") or 1.0)
    drop_pct = round(beta * 10, 1)
    pe = snap.get("pe")
    pe_str = f"P/E ratio of {pe:.1f}" if is_num(pe) else "valuation"
    return [
        QuizQuestion(
            category="Risk Tolerance",
            question=f"{name} has a Beta of {beta:.2f}. If the NIFTY drops 10%, historical price swings suggest this stock could pull back around {drop_pct}%. What is your risk management approach?",
            options={
                "A": "Panic and sell immediately.",
                "B": "Hold or accumulate in phases if the company's long-term thesis remains intact.",
                "C": "Borrow high-interest personal loans to double down.",
                "D": "Stop tracking the market completely."
            },
            correct_answer="B",
            explanation="Equities require discipline; position sizing according to Beta protects your capital from drawdowns."
        ),
        QuizQuestion(
            category="Business Model",
            question=f"Before committing money to {name} based on its current {pe_str}, what should you verify?",
            options={
                "A": "Whether trending posts on social messaging groups urge you to buy.",
                "B": "Core revenue quality, quarterly profit trend, and operating cash flows.",
                "C": "Only yesterday's percentage price move.",
                "D": "Anonymous chat tips promising guaranteed fast profits."
            },
            correct_answer="B",
            explanation="SEBI recommends analyzing core operational earnings and financial reports rather than speculative tips."
        ),
        QuizQuestion(
            category="Capital Allocation",
            question=f"What portion of your overall savings should you realistically allocate to a single direct stock like {name}?",
            options={
                "A": "Over 50% in an aggressive all-in bet.",
                "B": "Under 10-15% as part of a well-diversified portfolio basket.",
                "C": "100% using emergency funds.",
                "D": "All savings earmarked for next month's living expenses."
            },
            correct_answer="B",
            explanation="Diversification prevents single-company adverse events from impairing your entire financial future."
        ),
        QuizQuestion(
            category="Time Horizon",
            question=f"Direct equity investments in {name} should be matched with what realistic time horizon?",
            options={
                "A": "Emergency cash needed within 30 days.",
                "B": "Next month's household bills.",
                "C": "A medium to long-term horizon of 3 to 5+ years.",
                "D": "Intraday gambling on speculative rumors."
            },
            correct_answer="C",
            explanation="Equities carry volatility; compounding requires giving businesses sufficient time to execute."
        )
    ]


@app.post("/quiz/generate")
@app.post("/quiz")
def generate_quiz(req: QuizRequest) -> Dict[str, Any]:
    stock_ticker = req.target_stock or req.targetStock or req.ticker or req.symbol or "RELIANCE.NS"
    session_id = req.session_id or req.sessionId or secrets.token_hex(8)
    snap = fetch_stock_snapshot(normalize_ticker(stock_ticker))
    context = build_quiz_context(snap)

    req_lang = req.language or req.locale
    language = req_lang if req_lang in LANGUAGES else session_languages.get(session_id, DEFAULT_LANGUAGE)
    focus = ", ".join(random.sample(QUIZ_CATEGORIES, len(QUIZ_CATEGORIES)))
    system_prompt = (
        f"{QUIZ_PROMPT} Write every question, option and explanation in {LANGUAGES[language]}. "
        "Keep the JSON keys, the category names and the correct_answer letter in English. "
        f"Order the questions by these areas: {focus}. "
        f"Variation code {secrets.token_hex(3)}: make this quiz different from any earlier quiz."
    )

    questions: Optional[List[QuizQuestion]] = None
    for _ in range(QUIZ_LLM_ATTEMPTS):
        try:
            raw = call_colab_llm(f"{system_prompt}\n\nStock: {snap['name']}", context, system_prompt)
            questions = parse_quiz(raw)
            if questions:
                break
        except Exception as e:
            logger.warning("LLM generation attempt rejected: %s", e)

    if not questions:
        logger.info("Using dynamic fundamental quiz for %s", snap["ticker"])
        questions = generate_fallback_quiz(snap)

    quiz_list = [
        {
            "category": q.category,
            "question": q.question,
            "options": q.options,
            "correct_answer": q.correct_answer,
            "explanation": q.explanation,
        }
        for q in questions
    ]

    return {
        "session_id": session_id,
        "quiz_id": session_id,
        "ticker": snap["ticker"],
        "stock_name": snap["name"],
        "name": snap["name"],
        "topic": "Stock Safety & Valuation",
        "beta": snap["beta"],
        "language": language,
        "quiz": quiz_list,
        "questions": quiz_list,
    }


@app.post("/quiz/submit")
def submit_quiz(req: QuizSubmit) -> Dict[str, Any]:
    # Deprecated: quiz grading is now stateless and performed by the frontend against Neon DB.
    return {
        "detail": "Quiz grading is stateless. Evaluation is performed by the frontend application against persistent database records."
    }


@app.post("/voice/speak")
def voice_speak(req: SpeakRequest) -> Response:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Provide the text to speak.")
    language = session_languages.get(req.session_id, DEFAULT_LANGUAGE)
    return Response(content=speak(req.text, language), media_type="audio/mpeg")


@app.post("/voice/listen")
def voice_listen(session_id: str = Form(...), audio: UploadFile = File(...)) -> Dict[str, Any]:
    language = session_languages.get(session_id, DEFAULT_LANGUAGE)
    transcript = transcribe(audio)
    result = run_chat(session_id, transcript)
    return {
        "transcript": transcript,
        "language": language,
        "reply": result.reply,
        "detailed": result.detailed,
        "reply_audio_base64": base64.b64encode(speak(result.reply, language)).decode("ascii"),
        "audio_mime": "audio/mpeg",
        "rag_sources_used": result.rag_sources_used,
        "live_data_used": result.live_data_used,
    }


@app.post("/analyze-fomo")
def analyze_fomo(req: FomoRequest) -> Dict[str, Any]:
    ticker = req.ticker or extract_ticker(req.message)
    return evaluate_fomo(req.message, ticker, req.demo_mode)


@app.get("/market-alerts")
def market_alerts(triggered_only: bool = False) -> Dict[str, Any]:
    alerts = list(market_state["alerts"].values())
    if triggered_only:
        alerts = [a for a in alerts if a["triggered"]]
    return {
        "checked_at": market_state["checked_at"],
        "threshold_pct": DROP_THRESHOLD,
        "simulated": SIMULATE_STREAM,
        "alerts": alerts,
    }


@app.post("/scan-image")
def scan_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 5 MB limit.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not decode the image.")
    try:
        return analyze_image(image)
    except Exception:
        logger.exception("OCR pipeline failed")
        raise HTTPException(status_code=500, detail="OCR processing failed.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000)
