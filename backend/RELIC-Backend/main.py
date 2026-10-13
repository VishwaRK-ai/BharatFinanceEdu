import os
import re
import json
import traceback
from typing import Any, Dict, List, Optional

import gradio as gr

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# =====================================================================
# GLOBAL CONFIGURATION & STOCK LOOKUP (Defined at top to avoid NameError)
# =====================================================================
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


def get_stock_display_name(ticker: Optional[str]) -> str:
    """Safe lookup for human-readable stock name, guaranteed never to throw."""
    if not ticker or not isinstance(ticker, str):
        return "Reliance Industries"
    clean = ticker.upper().strip()
    if clean in STOCK_NAMES:
        return STOCK_NAMES[clean]
    base = clean.replace(".NS", "").replace(".BO", "")
    return base.title() if base else "Reliance Industries"


# =====================================================================
# 1. CHROMADB VECTORSTORE INITIALIZATION
# =====================================================================
CHROMA_PATH = "/home/azureuser/backend/RELIC-Backend/chroma_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
print(f"Initializing embedding model '{EMBEDDING_MODEL}'...")
embeddings = None
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
except Exception as e_emb:
    print(f"Warning: HuggingFaceEmbeddings could not be loaded: {e_emb}")

print(f"Loading Chroma vectorstore from {CHROMA_PATH}...")
vectorstore = None
if embeddings is not None:
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        print("✅ Chroma vectorstore initialized successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize Chroma vectorstore from {CHROMA_PATH}: {e}")
        # Local fallback for development / non-Azure environments
        if os.path.exists("./chroma_db"):
            vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        elif os.path.exists("backend/RELIC-Backend/chroma_db"):
            vectorstore = Chroma(persist_directory="backend/RELIC-Backend/chroma_db", embedding_function=embeddings)
        else:
            vectorstore = None

# =====================================================================
# 2. MODEL LOADING (LLAMACPP)
# =====================================================================
from langchain_community.llms import LlamaCpp
print('Loading 4.4GB GGUF model into CPU RAM...')
llm = LlamaCpp(
    model_path='Qwen2.5-7B-Instruct.Q4_K_M.gguf',
    n_ctx=4096,
    max_tokens=1024,
    n_threads=4,
)




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
    raw_text: str,
    target_stock: str,
    session_id: str
) -> Optional[Dict[str, Any]]:
    """Extracts, cleans, and validates structured quiz JSON conforming to frontend normalizeQuizStart."""
    if not raw_text or not isinstance(raw_text, str):
        return None

    cleaned = raw_text.strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        # Search for outermost JSON object { ... }
        match_obj = re.search(r"\{[\s\S]*\}", cleaned)
        if match_obj:
            try:
                parsed = json.loads(match_obj.group(0))
            except Exception:
                pass

        # If still None, search for JSON array [ ... ]
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

    # Find list of questions
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


def generate_text(
    prompt_text: str,
    max_tokens: int = 256,
    system_instruction: Optional[str] = None,
    temperature: float = 0.3
) -> str:
    """
    Generate LLM completion from prompt using LlamaCpp.
    """
    global llm

    if system_instruction is None:
        system_instruction = (
            "You are BharatFinanceEdu, an expert Indian financial educator. "
            "Explain concepts simply, using clear analogies and dispelling misconceptions. "
            "Base your answers on the provided context. If the user acts impulsively, warn them based on SEBI rules."
        )

    if llm is not None:
        try:
            full_prompt = f"{system_instruction}\n\n{prompt_text}"
            print(f"Invoking LlamaCpp instance with prompt (length {len(full_prompt)})...")

            # Try modern LangChain invoke() first
            if hasattr(llm, "invoke"):
                res = llm.invoke(full_prompt)
                if hasattr(res, "content"):
                    return str(res.content).strip()
                return str(res).strip()
            # Fall back to direct callable: llm(prompt)
            elif callable(llm):
                res = llm(full_prompt)
                if isinstance(res, dict) and "choices" in res:
                    return str(res["choices"][0]["text"]).strip()
                return str(res).strip()
            else:
                raise AttributeError("llm instance has neither invoke method nor is callable")
        except Exception as e:
            print(f"Error during LlamaCpp generation: {e}")
            traceback.print_exc()
            raise e

    print("Error: LlamaCpp (`llm`) is not initialized.")
    raise RuntimeError("Model is not initialized.")


# =====================================================================
# 3. PREDICTION FUNCTION WITH RAG SIMILARITY SEARCH
# =====================================================================
def predict(prompt: Any, max_tokens: float = 256) -> str:
    """
    Main prediction endpoint called via Gradio /api/predict.
    Handles both structured Quiz generation and conversational assistant queries.
    Performs similarity search against ChromaDB using target_stock (k=3),
    injects context into the prompt, and returns schema-conforming responses.
    """
    try:
        target_stock = "RELIANCE.NS"
        user_text = ""
        session_id = "default_session"
        is_quiz = False

        # 1. Parse prompt and detect request type
        parsed = None
        if isinstance(prompt, dict):
            parsed = prompt
        elif isinstance(prompt, str):
            try:
                val = json.loads(prompt)
                if isinstance(val, dict):
                    parsed = val
            except Exception:
                parsed = None

        if parsed is not None:
            # Safely extract target stock with guaranteed fallback
            raw_stock = (
                parsed.get("target_stock")
                or parsed.get("targetStock")
                or parsed.get("ticker")
                or parsed.get("stock_name")
                or parsed.get("symbol")
            )
            if raw_stock and isinstance(raw_stock, str) and raw_stock.strip():
                target_stock = raw_stock.strip()
            else:
                target_stock = "RELIANCE.NS"

            session_id = str(parsed.get("session_id") or parsed.get("sessionId") or "default_session")

            # Extract message / user prompt cleanly
            user_text = (
                parsed.get("message")
                or parsed.get("prompt")
                or parsed.get("text")
                or parsed.get("query")
                or ""
            )
            if not isinstance(user_text, str):
                user_text = str(user_text) if user_text is not None else ""

            # Check if request is quiz action
            action = str(parsed.get("action", "")).lower()
            if action in ("quiz", "quiz/generate", "generate_quiz"):
                is_quiz = True
            elif ("target_stock" in parsed or "targetStock" in parsed) and not user_text:
                is_quiz = True
            elif "target_stock" in parsed and "quiz" in user_text.lower():
                is_quiz = True

            # If user_text is empty, serialize dictionary
            if not user_text:
                user_text = json.dumps(parsed)
        else:
            user_text = str(prompt) if prompt is not None else ""
            if "quiz" in user_text.lower() and ("generate" in user_text.lower() or "target_stock" in user_text.lower()):
                is_quiz = True

            # Heuristic extraction of stock ticker if not in JSON
            known_stocks = [
                "RELIANCE", "TATAMOTORS", "TATA MOTORS", "ZOMATO",
                "SUZLON", "HDFCBANK", "HDFC", "INFY", "INFOSYS", "ITC", "SBIN", "SBI", "TCS"
            ]
            for s in known_stocks:
                if re.search(rf"\b{s}\b", user_text, re.IGNORECASE):
                    target_stock = f"{s.replace(' ', '')}.NS" if not s.endswith(".NS") else s
                    break

        if not target_stock or not isinstance(target_stock, str):
            target_stock = "RELIANCE.NS"

        stock_name = get_stock_display_name(target_stock)

        # 2. Similarity search using ChromaDB (k=3)
        context_str = ""
        if vectorstore is not None:
            try:
                if is_quiz:
                    search_query = f"{stock_name} ({target_stock}) investor suitability, volatility risk and SEBI guidelines"
                else:
                    search_query = f"{user_text} {target_stock} investment risk profile and SEBI guidelines".strip()

                docs = vectorstore.similarity_search(search_query, k=3)
                chunks = [doc.page_content.strip() for doc in docs if getattr(doc, "page_content", None)]
                context_str = "\n\n".join(chunks)
            except Exception as e:
                print(f"Similarity search error: {e}")
                traceback.print_exc()

        # 3. QUIZ GENERATION BRANCH
        if is_quiz:
            print(f"🎯 Quiz generation requested for {stock_name} ({target_stock})")
            quiz_system_instruction = (
                "You are BharatFinanceEdu, an expert SEBI-aligned Indian financial educator and investor readiness examiner. "
                "You MUST output ONLY a valid, parseable JSON object matching the required schema. "
                "Do NOT include any markdown code blocks, backticks (like ```json), or conversational prose. "
                "Return pure JSON only."
            )

            quiz_prompt = (
                f"Generate a 4-question investor readiness and suitability quiz for {stock_name} ({target_stock}).\n\n"
                f"SEBI/RBI Reference Context:\n{context_str}\n\n"
                "Generate 4 multiple-choice questions covering these 4 categories:\n"
                "1. 'Risk Tolerance': downside risk, volatility, Beta, and drawdowns.\n"
                "2. 'Valuation': P/E, 52-week range, and margin of safety.\n"
                "3. 'Business Model': cash flows, quarterly profit trends, and operational revenue.\n"
                "4. 'Capital Allocation': position sizing (keeping single stock under 10-15%), diversification, and emergency funds.\n\n"
                "Output strictly a JSON object conforming to this exact schema:\n"
                "{\n"
                f'  "session_id": "{session_id}",\n'
                f'  "ticker": "{target_stock}",\n'
                f'  "stock_name": "{stock_name}",\n'
                '  "topic": "Investor Readiness & Risk Management",\n'
                '  "quiz": [\n'
                '    {\n'
                '      "category": "Risk Tolerance",\n'
                '      "question": "Question text quoting risk metrics?",\n'
                '      "options": {\n'
                '        "A": "Option text A",\n'
                '        "B": "Option text B",\n'
                '        "C": "Option text C",\n'
                '        "D": "Option text D"\n'
                '      },\n'
                '      "correct_answer": "B",\n'
                '      "explanation": "One clear sentence explaining the SEBI-aligned correct choice."\n'
                '    }\n'
                '  ]\n'
                "}\n\n"
                "STRICT RULES:\n"
                "- Exactly 4 questions, each with options A, B, C, D.\n"
                "- 'correct_answer' must be one of 'A', 'B', 'C', or 'D'.\n"
                "- Do not output markdown fences or backticks. Return valid JSON only."
            )

            try:
                raw_llm_out = generate_text(
                    prompt_text=quiz_prompt,
                    max_tokens=max(int(max_tokens), 1024),
                    system_instruction=quiz_system_instruction,
                    temperature=0.2
                )
                print(f"Raw LLM output (first 150 chars): {raw_llm_out[:150]}")
                validated_quiz = extract_and_validate_quiz(raw_llm_out, target_stock, session_id)
                if validated_quiz:
                    print("✅ Successfully extracted and validated LLM quiz.")
                    return json.dumps(validated_quiz)
            except Exception as e:
                print(f"Error during Quiz LLM generation: {e}")
                traceback.print_exc()

            print("⚠️ LLM output could not be parsed as valid quiz schema. Using deterministic fallback quiz.")
            fallback_quiz = generate_fallback_quiz(target_stock, session_id)
            return json.dumps(fallback_quiz)

        # 4. CHAT / ASSISTANT GENERATION BRANCH
        chat_system_instruction = (
            "You are BharatFinanceEdu, an expert Indian financial educator. "
            "Explain concepts simply, using clear analogies and dispelling misconceptions. "
            "Base your answers on the provided context. If the user acts impulsively, warn them based on SEBI rules."
        )
        if context_str:
            final_prompt = (
                f"Reference Context (SEBI/RBI Guidelines from ChromaDB):\n{context_str}\n\n"
                f"Target Stock: {target_stock}\n\n"
                f"User Prompt:\n{user_text}"
            )
        else:
            final_prompt = f"Target Stock: {target_stock}\n\nUser Prompt:\n{user_text}"

        try:
            return generate_text(final_prompt, int(max_tokens), system_instruction=chat_system_instruction)
        except Exception as e:
            print(f"Error during Chat LLM generation: {e}")
            traceback.print_exc()
            return "I am currently connecting to market intelligence servers. Please ask your question again in a moment."

    except Exception as e:
        print(f"Error inside predict: {e}")
        traceback.print_exc()
        return "I am currently connecting to market intelligence servers. Please ask your question again in a moment."



# =====================================================================
# 4. GRADIO INTERFACE SETUP & LAUNCH
# =====================================================================
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Prompt", lines=5),
        gr.Slider(minimum=64, maximum=2048, value=256, step=64, label="Max Tokens"),
    ],
    outputs=gr.Textbox(label="Response", lines=5),
    title="BharatFinanceEdu AI Inference API",
)

# Export FastAPI app for Render uvicorn compatibility if imported
try:
    from main1 import app
except Exception:
    app = None

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=("admin", "BharatFinance2026!"),
    )
