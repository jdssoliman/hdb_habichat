# hdb_backend.py
"""
Core logic (UI-free):
- Key rotation (Google/Gemini)
- HF Inference (Llama 3)
- Payslip parsing with Gemini (PDF/PNG/JPG)
- DB loader
- General HDB Q&A
- Salary-only decision (ACCEPT/REJECT) based on payslip vs declared income
"""

from __future__ import annotations

import os
import re
import time
import json
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st
from huggingface_hub import InferenceClient
import google.generativeai as genai

# -------------------------
# Constants & Regex
# -------------------------
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
IC_REGEX = re.compile(r"\b([STFG]\d{7}[A-Z])\b")
_NUM_RX = re.compile(r"[-+]?\d{1,3}(?:[, ]\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")

QUESTION_HINTS = [
    "eligibility", "apply", "application", "hfe", "hfe letter", "loan",
    "income ceiling", "grants", "bto", "resale", "ec", "executive condominium",
    "cpf", "cpf oa", "tenure", "interest", "msr", "valuation", "lease",
    "hdb flat portal", "documents", "payslip", "noa", "notice of assessment",
    "age", "citizen", "pr", "private property", "cooling period",
    "mismatch", "appeal", "reject", "accept", "bank loan", "hdb loan"
]

# -------------------------
# Google API key rotation
# -------------------------
def _load_google_keys() -> list[str]:
    keys: list[str] = []
    try:
        keys = list(st.secrets.get("GOOGLE_API_KEYS", []))  # type: ignore[attr-defined]
    except Exception:
        pass
    if not keys:
        env_keys = os.environ.get("GOOGLE_API_KEYS", "")
        if env_keys:
            keys = [k.strip() for k in env_keys.split(",") if k.strip()]
    return keys

GOOGLE_KEYS = _load_google_keys()

def init_key_index():
    if "gapi_idx" not in st.session_state:
        st.session_state.gapi_idx = 0  # round-robin pointer

def current_google_key() -> Optional[str]:
    if GOOGLE_KEYS:
        return GOOGLE_KEYS[st.session_state.gapi_idx % len(GOOGLE_KEYS)]
    # fallback single key
    try:
        key = st.secrets.get("GOOGLE_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        key = None
    key = key or os.environ.get("GOOGLE_API_KEY")
    return key

def advance_key():
    if GOOGLE_KEYS:
        st.session_state.gapi_idx = (st.session_state.gapi_idx + 1) % len(GOOGLE_KEYS)

def has_any_google_key() -> bool:
    if current_google_key():
        return True
    try:
        _ = st.secrets["GOOGLE_API_KEYS"]  # type: ignore[index]
        return True
    except Exception:
        return bool(os.environ.get("GOOGLE_API_KEYS"))

# -------------------------
# HF Inference (Hosted, Chat Completions)
# -------------------------
@st.cache_resource(show_spinner=True)
def get_hf_client() -> InferenceClient:
    token = None
    try:
        token = st.secrets.get("HUGGINGFACE_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        pass
    token = token or os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        raise RuntimeError("Missing Hugging Face token. Set HUGGINGFACE_API_KEY.")
    return InferenceClient(token=token)

def _extract_choice_text(resp) -> str:
    choice = resp.choices[0]
    msg = getattr(choice, "message", getattr(choice, "delta", None))
    if msg is None:
        msg = choice.get("message", choice.get("delta", {}))  # type: ignore[union-attr]
    content = getattr(msg, "content", None)
    if content is None:
        content = msg.get("content", "")  # type: ignore[union-attr]
    return content or ""

def llama3_chat_safe(
    messages: List[Dict[str, str]],
    max_tokens: int = 380,
    temperature: float = 0.2,
    top_p: float = 0.9,
    retries: int = 3,
) -> str:
    client = get_hf_client()
    last_err = None
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return _extract_choice_text(resp)
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (i + 1))
    return f"(LLM error after {retries} retries: {last_err})"

# -------------------------
# Gemini helpers (payslip OCR/parse)
# -------------------------
def _to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = _NUM_RX.search(str(x))
    if not m:
        return None
    s = m.group(0).replace(",", "").replace(" ", "")
    try:
        return float(s)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_gemini_model_for_key(key: str) -> genai.GenerativeModel:
    if not key:
        raise RuntimeError("Missing Google API key.")
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-2.5-flash")

def gemini_generate_with_rotation(parts: list, max_attempts: int | None = None):
    attempts = max_attempts or max(1, len(GOOGLE_KEYS) or [1][0])
    last_err = None
    for i in range(attempts):
        key = current_google_key()
        if not key:
            raise RuntimeError("No Google API key available. Configure GOOGLE_API_KEYS or GOOGLE_API_KEY.")
        try:
            model = get_gemini_model_for_key(key)
            return model.generate_content(parts)
        except Exception as e:
            msg = str(e).lower()
            last_err = e
            if any(t in msg for t in ["429", "quota", "rate", "exhaust"]):
                advance_key()
                time.sleep(0.6 + 0.2 * i)
                continue
            else:
                raise
    raise RuntimeError(f"All Google API keys failed or hit quota. Last error: {last_err}")

def parse_payslip(uploaded_file) -> Optional[float]:
    """
    Uses Gemini to parse a payslip (PDF/PNG/JPG) and returns Basic Salary as float.
    """
    prompt = """
    You are a strict JSON document parser.
    Task: From this payslip, extract ONLY these fields:
      - Employee Name
      - Employer Name
      - Basic Salary (monthly base pay)
      - Deductions
      - Gross Income
      - Net Income
    Rules:
      - Return ONLY valid JSON with EXACT keys:
        employee_name, employer_name, basic_salary, deductions, gross_income, net_income
      - If a field is missing or not stated, set it to null.
      - "basic_salary" must be the monthly base pay (exclude allowances, OT, bonuses, CPF/SSS, etc.).
    """
    file_bytes = uploaded_file.read()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        mime_type = "application/pdf"
    elif suffix == ".png":
        mime_type = "image/png"
    else:
        mime_type = "image/jpeg"  # jpg/jpeg
    resp = gemini_generate_with_rotation([
        prompt,
        {"mime_type": mime_type, "data": file_bytes}
    ])
    text = (getattr(resp, "text", "") or "").strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        text = parts[1] if len(parts) > 1 else text
        if text.lstrip().lower().startswith("json"):
            text = text.split("\n", 1)[1].strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _to_float(obj.get("basic_salary"))

# -------------------------
# Data access
# -------------------------
@st.cache_data
def load_db() -> pd.DataFrame:
    df = pd.read_csv("/data/mock_db.csv")  # ensure .csv
    return df

# -------------------------
# Utility & Q&A helpers
# -------------------------
def mask_ic(ic: str) -> str:
    if not ic or len(ic) < 3:
        return ic
    return ic[:2] + "*****" + ic[-1:]

def looks_like_question(txt: str) -> bool:
    t = (txt or "").strip().lower()
    if not t:
        return False
    if "?" in t:
        return True
    return any(k in t for k in QUESTION_HINTS)

def hdb_general_answer(user_text: str, profile_hint: dict | None = None) -> str:
    sys = (
        "You are a helpful assistant for Singapore HDB housing loans. "
        "Answer briefly (3–6 bullets or 1–2 short paragraphs). "
        "Be practical, non-legal, and avoid guarantees or exact policy numbers/rates that change. "
        "Explain terms like HFE, MSR, TDSR, income ceilings, grants, tenure, and steps if asked. "
        "When in doubt, advise checking the HDB Flat Portal for the latest."
    )
    msgs = [{"role": "system", "content": sys}]
    if profile_hint:
        msgs.append({"role": "user", "content": f"Context (may help): {profile_hint}"})
    msgs.append({"role": "user", "content": (user_text or "").strip()})
    out = llama3_chat_safe(msgs, max_tokens=300, temperature=0.2, top_p=0.9)
    if not out or out.startswith("(LLM error"):
        return "Here’s a quick overview. Criteria may change; please check the HDB Flat Portal for the latest details."
    s = " ".join(out.split())
    if "HDB Flat Portal" not in s and "HFE" not in s:
        s += " For the latest policy details, please check the HDB Flat Portal."
    return s

# -------------------------
# Salary-only decision
# -------------------------
def salary_only_decision(row: pd.Series, payslip_income: Optional[float]) -> dict:
    """
    Decide ACCEPT/REJECT using only payslip salary vs DB 'Declared Income'.
    Rules:
      - If payslip not readable -> REJECT
      - If declared income missing/zero -> REJECT
      - If |payslip - declared| / declared > 0.10 -> REJECT
      - else -> ACCEPT
    """
    reasons: list[str] = []

    if payslip_income is None:
        reasons.append("Payslip income could not be read. Please re-upload a clearer payslip.")
        return {"decision": "REJECT", "reasons": reasons}

    declared = float(row.get("Declared Income", 0) or 0)
    if declared <= 0:
        reasons.append("No declared income on record to compare with.")
        return {"decision": "REJECT", "reasons": reasons, "declared_income": declared, "payslip_income": payslip_income}

    diff_ratio = abs(payslip_income - declared) / declared
    if diff_ratio > 0.10:
        reasons.append(
            f"Payslip income (S${payslip_income:,.0f}) does not match declared income (S${declared:,.0f}) beyond 10%."
        )
        return {"decision": "REJECT", "reasons": reasons, "declared_income": declared, "payslip_income": payslip_income}

    return {"decision": "ACCEPT", "reasons": [], "declared_income": declared, "payslip_income": payslip_income}


def llm_explain_hosted_json(decision: str, reasons: List[str], profile: Dict) -> Dict:
    import re as _re, json as _json
    sys_msg = (
        "You are a helpful HDB assistant.\n"
        "Respond ONLY with a single valid JSON object. No markdown.\n"
        "Schema:\n"
        "{\n"
        '  \"summary\": \"string (1-2 sentences, plain English)\",\n'
        '  \"tone\": \"accept\" | \"reject\",\n'
        '  \"next_steps\": [\"string\", ...],\n'
        '  \"notes\": \"string or null\"\n'
        "}\n"
        "If decision is ACCEPT, tone MUST be accept; else reject.\n"
        "Be specific and practical; avoid legalese."
    )
    ex_accept_user = (
        "Decision: ACCEPT\n"
        "Reasons: None\n"
        "Profile: {\"Name\":\"Alice Tan\",\"Citizenship\":\"Singapore Citizen\",\"Marital Status\":\"Single\",\"Declared Income\":4800,\"Payslip Income\":4800,\"Household Type\":\"Single\"}\n"
        "Write the JSON."
    )
    ex_accept_assistant = {
        "summary": "You appear eligible based on income and citizenship. This is a preliminary check—please proceed with formal verification.",
        "tone": "accept",
        "next_steps": [
            "Apply for an HFE letter via the HDB Flat Portal",
            "Prepare the latest 3 months of payslips or NOA",
            "Verify personal and household particulars in the application",
            "Review loan ceiling and tenure before submission"
        ],
        "notes": "Eligibility may change after full document checks."
    }
    ex_reject_user = (
        "Decision: REJECT\n"
        "Reasons: Payslip income (S$9,500) does not match declared income (S$7,000) beyond 10%.\n"
        "Profile: {\"Name\":\"Peter Lim\",\"Citizenship\":\"Singapore Citizen\",\"Marital Status\":\"Married\",\"Declared Income\":7000,\"Payslip Income\":9500,\"Household Type\":\"Family\"}\n"
        "Write the JSON."
    )
    ex_reject_assistant = {
        "summary": "You do not currently meet HDB loan criteria due to a large mismatch between declared and payslip income.",
        "tone": "reject",
        "next_steps": [
            "Align declared income with verifiable documents (e.g., payslips or NOA)",
            "Update income records and re-check eligibility after corrections",
            "Consider bank financing options meanwhile"
        ],
        "notes": "If documentation clarifies the income variance, the outcome may differ."
    }
    user_msg = (
        f"Decision: {decision}\n"
        f"Reasons: {', '.join(reasons) if reasons else 'None'}\n"
        f"Profile: {profile}\n"
        "Write the JSON."
    )
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": ex_accept_user},
        {"role": "assistant", "content": _json.dumps(ex_accept_assistant, ensure_ascii=False)},
        {"role": "user", "content": ex_reject_user},
        {"role": "assistant", "content": _json.dumps(ex_reject_assistant, ensure_ascii=False)},
        {"role": "user", "content": user_msg},
    ]
    raw = llama3_chat_safe(messages, max_tokens=220, temperature=0.2, top_p=0.9)
    if raw.startswith("(LLM error"):
        return {"summary": raw, "tone": decision.lower(), "next_steps": [], "notes": None}
    s = raw.strip()
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) > 1 else s
    m = _re.search(r"\{.*\}", s, _re.S)
    if not m:
        return {"summary": s[:180], "tone": decision.lower(), "next_steps": [], "notes": None}
    try:
        obj = _json.loads(m.group(0))
    except Exception:
        return {"summary": s[:180], "tone": decision.lower(), "next_steps": [], "notes": None}
    obj.setdefault("summary", "")
    obj.setdefault("tone", decision.lower())
    obj.setdefault("next_steps", [])
    obj.setdefault("notes", None)
    obj["tone"] = "accept" if decision.upper() == "ACCEPT" else "reject"
    if not isinstance(obj["next_steps"], list):
        obj["next_steps"] = []
    obj["next_steps"] = [str(x).strip() for x in obj["next_steps"] if str(x).strip()][:5]
    if len(obj["next_steps"]) < 2:
        if obj["tone"] == "accept":
            obj["next_steps"] += [
                "Apply for an HFE letter via the HDB Flat Portal",
                "Prepare the latest 3 months of payslips or NOA"
            ]
        else:
            obj["next_steps"] += [
                "Resolve the listed issues (e.g., unreadable payslip or income mismatch)",
                "Re-check after updating documents"
            ]
        obj["next_steps"] = obj["next_steps"][:5]
    obj["summary"] = " ".join(str(obj["summary"]).split())[:400]
    if obj["notes"] is not None:
        obj["notes"] = " ".join(str(obj["notes"]).split())[:200]
    return obj
