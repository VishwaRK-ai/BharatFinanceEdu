import os
import re
import json
from typing import Any, Optional

import torch
import gradio as gr

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# =====================================================================
# 1. CHROMADB VECTORSTORE INITIALIZATION
# =====================================================================
CHROMA_PATH = "/home/azureuser/backend/RELIC-Backend/chroma_db"
print(f"Initializing embedding model 'all-MiniLM-L6-v2'...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print(f"Loading Chroma vectorstore from {CHROMA_PATH}...")
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
# 2. MODEL LOADING (UNSLOTH / TRANSFORMERS)
# =====================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Serving on device: {device}")

model = None
tokenizer = None

try:
    from unsloth import FastLanguageModel
    print("Loading model using Unsloth FastLanguageModel...")
    model_path = "/home/azureuser/backend/RELIC-Backend/bharat_finance_model"
    if not os.path.exists(model_path):
        model_path = "bharat_finance_model" if os.path.exists("bharat_finance_model") else "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("✅ Fine-tuned Unsloth model loaded successfully.")
except Exception as e_unsloth:
    print(f"Unsloth load bypassed/failed: {e_unsloth}. Trying transformers AutoModelForCausalLM...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = "/home/azureuser/backend/RELIC-Backend/bharat_finance_model"
        if not os.path.exists(model_path):
            model_path = "bharat_finance_model" if os.path.exists("bharat_finance_model") else "Qwen/Qwen2.5-7B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        print("✅ Transformers model loaded successfully.")
    except Exception as e_trans:
        print(f"⚠️ Transformers model load warning: {e_trans}")
        model = None
        tokenizer = None


def generate_text(prompt_text: str, max_tokens: int = 256) -> str:
    """Generate LLM completion from prompt."""
    if model is None or tokenizer is None:
        return "Model is not initialized."
    try:
        system_instruction = (
            "You are BharatFinanceEdu, an expert Indian financial educator. "
            "Explain concepts simply, using clear analogies and dispelling misconceptions. "
            "Base your answers on the provided context. If the user acts impulsively, warn them based on SEBI rules."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_text},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=int(max_tokens),
                temperature=0.3,
                use_cache=True,
            )
        response = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
        return response.strip()
    except Exception as e:
        print(f"Generation error: {e}")
        return f"Error during generation: {e}"


# =====================================================================
# 3. PREDICTION FUNCTION WITH RAG SIMILARITY SEARCH
# =====================================================================
def predict(prompt: Any, max_tokens: float = 256) -> str:
    """
    Main prediction endpoint called via Gradio /api/predict.
    Performs similarity search against ChromaDB using target_stock (k=3),
    injects context into the prompt, and generates the LLM response.
    """
    target_stock = "RELIANCE.NS"
    user_text = ""

    # 1. Parse prompt and extract target_stock
    if isinstance(prompt, dict):
        target_stock = (
            prompt.get("target_stock")
            or prompt.get("targetStock")
            or prompt.get("ticker")
            or prompt.get("stock_name")
            or prompt.get("symbol")
            or "RELIANCE.NS"
        )
        user_text = prompt.get("message") or prompt.get("prompt") or json.dumps(prompt)
    elif isinstance(prompt, str):
        try:
            parsed = json.loads(prompt)
            if isinstance(parsed, dict):
                target_stock = (
                    parsed.get("target_stock")
                    or parsed.get("targetStock")
                    or parsed.get("ticker")
                    or parsed.get("stock_name")
                    or parsed.get("symbol")
                    or "RELIANCE.NS"
                )
                user_text = parsed.get("message") or parsed.get("prompt") or prompt
            else:
                user_text = prompt
        except Exception:
            user_text = prompt

        # Heuristic extraction of stock ticker if not in JSON
        if not target_stock or target_stock == "RELIANCE.NS":
            known_stocks = [
                "RELIANCE", "TATAMOTORS", "TATA MOTORS", "ZOMATO",
                "SUZLON", "HDFCBANK", "HDFC", "INFY", "INFOSYS", "ITC", "SBIN", "SBI"
            ]
            for s in known_stocks:
                if re.search(rf"\b{s}\b", user_text, re.IGNORECASE):
                    target_stock = f"{s.replace(' ', '')}.NS" if not s.endswith(".NS") else s
                    break
    else:
        user_text = str(prompt)

    # 2. Similarity search using target_stock (k=3)
    context_str = ""
    if vectorstore is not None:
        try:
            search_query = f"{target_stock} investment risk profile and SEBI guidelines" if target_stock else user_text
            docs = vectorstore.similarity_search(search_query, k=3)
            chunks = [doc.page_content.strip() for doc in docs if getattr(doc, "page_content", None)]
            context_str = "\n\n".join(chunks)
        except Exception as e:
            print(f"Similarity search error: {e}")

    # 3. Format retrieved documents into context string and inject into final prompt
    if context_str:
        final_prompt = (
            f"Reference Context (SEBI/RBI Guidelines from ChromaDB):\n{context_str}\n\n"
            f"Target Stock: {target_stock}\n\n"
            f"User Prompt:\n{user_text}"
        )
    else:
        final_prompt = f"Target Stock: {target_stock}\n\nUser Prompt:\n{user_text}"

    # 4. Generate and return the response
    return generate_text(final_prompt, int(max_tokens))


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
