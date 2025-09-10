"""
🏠 HDB Loan Chatbot – Llama 3 8B (Hugging Face Hosted) + Gemini Payslip Parsing

Quick start:
  pip install --upgrade streamlit pandas PyPDF2 "huggingface_hub>=0.28.0" google-generativeai
  export HUGGINGFACE_API_KEY=hf_...
  export GOOGLE_API_KEY=AIza...
  streamlit run hdb_loan_chatbot_llama3_streamlit.py

Notes:
- Uses HF Chat Completions (task=conversational) for meta-llama/Meta-Llama-3-8B-Instruct
- Uses Gemini (no Client object needed) to parse payslip PDFs
- Requires fine-grained HF token & a Gemini API key
"""

import os
import io
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

import json
import pandas as pd
import streamlit as st

# -------------------------
# Google API key rotation
# -------------------------
def _load_google_keys() -> list[str]:
    keys = []
    try:
        keys = list(st.secrets.get("GOOGLE_API_KEYS", []))  # type: ignore[attr-defined]
    except Exception:
        pass
    if not keys:
        # Fallback to env var (comma-separated) or hardcoded
        env_keys = os.environ.get("GOOGLE_API_KEYS", "")
        if env_keys:
            keys = [k.strip() for k in env_keys.split(",") if k.strip()]
    # LAST resort: hardcode here (not recommended)
    # keys = keys or ["AIza...1", "AIza...2"]
    return keys

GOOGLE_KEYS = _load_google_keys()
if not GOOGLE_KEYS:
    st.warning("No GOOGLE_API_KEYS configured. Falling back to single GOOGLE_API_KEY (if set).")

if "gapi_idx" not in st.session_state:
    st.session_state.gapi_idx = 0  # round-robin pointer

def _current_google_key() -> Optional[str]:
    if GOOGLE_KEYS:
        return GOOGLE_KEYS[st.session_state.gapi_idx % len(GOOGLE_KEYS)]
    # single-key fallback
    try:
        key = st.secrets.get("GOOGLE_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        key = None
    key = key or os.environ.get("GOOGLE_API_KEY")
    return key

def _advance_key():
    if GOOGLE_KEYS:
        st.session_state.gapi_idx = (st.session_state.gapi_idx + 1) % len(GOOGLE_KEYS)


# =========================
# HF Inference API (Hosted, Chat Completions)
# =========================
from huggingface_hub import InferenceClient

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # accept license on HF

@st.cache_resource(show_spinner=True)
def get_hf_client() -> InferenceClient:
    """
    Returns a cached HF InferenceClient. Looks for token in:
      - st.secrets["HUGGINGFACE_API_KEY"] or env HUGGINGFACE_API_KEY
    """
    token = None
    try:
        token = st.secrets.get("HUGGINGFACE_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        pass
    token = token or os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        raise RuntimeError("Missing Hugging Face token. Set HUGGINGFACE_API_KEY in env or st.secrets.")
    return InferenceClient(token=token)

def _extract_choice_text(resp) -> str:
    """Robustly extract the first choice content from chat.completions.create response."""
    choice = resp.choices[0]
    # Newer clients: object with attributes; older: dicts
    msg = getattr(choice, "message", getattr(choice, "delta", None))
    if msg is None:
        msg = choice.get("message", choice.get("delta", {}))  # type: ignore[union-attr]
    content = getattr(msg, "content", None)
    if content is None:
        content = msg.get("content", "")  # type: ignore[union-attr]
    return content or ""

def llama3_chat_safe(
    messages: List[Dict[str, str]],
    max_tokens: int = 200,
    temperature: float = 0.2,
    top_p: float = 0.9,
    retries: int = 3,
) -> str:
    """
    Non-streaming chat call with retries.
    Returns text or an error string prefixed with (LLM error: ...)
    """
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

# (Optional) streaming helper (unused by default, kept for future)
def llama3_chat_stream(
    messages: List[Dict[str, str]],
    max_tokens: int = 160,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    client = get_hf_client()
    stream = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )
    for chunk in stream:
        try:
            delta_obj = chunk.choices[0].delta
            delta_text = getattr(delta_obj, "content", None)
            if delta_text is None:
                delta_text = delta_obj.get("content", "")  # type: ignore[union-attr]
            if delta_text:
                yield delta_text
        except Exception:
            pass

# =========================
# Payslip parsing (Gemini → returns float basic salary)
# =========================
import google.generativeai as genai

_NUM_RX = re.compile(r"[-+]?\d{1,3}(?:[, ]\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")

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
    """Caches a GenerativeModel instance per API key."""
    if not key:
        raise RuntimeError("Missing Google API key.")
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-2.5-flash")

import time

def gemini_generate_with_rotation(parts: list, max_attempts: int = None):
    """
    Call Gemini.generate_content with key rotation.
    Tries current key; on 429/quota/transient error, advances to next key and retries.
    """
    attempts = max_attempts or max(1, len(GOOGLE_KEYS))  # try each key once by default
    last_err = None

    for i in range(attempts):
        key = _current_google_key()
        if not key:
            raise RuntimeError("No Google API key available. Configure GOOGLE_API_KEYS or GOOGLE_API_KEY.")

        try:
            model = get_gemini_model_for_key(key)
            return model.generate_content(parts)

        except Exception as e:
            msg = str(e).lower()
            last_err = e
            # Common quota / rate limit signals: 429, quota, rate, exhaust(ed), resource exhausted
            if any(t in msg for t in ["429", "quota", "rate", "exhaust"]):
                # rotate to next key and try again
                _advance_key()
                # small backoff
                time.sleep(0.6 + 0.2 * i)
                continue
            else:
                # Non-quota error -> bubble up immediately
                raise

    # If we exhausted attempts:
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

    # ---- use the rotating caller here
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


# =========================
# Eligibility rules (demo)
# =========================
INCOME_CEILING = {"Family": 14000, "Extended Family": 21000, "Single": 7000}

def check_eligibility(
    row: pd.Series,
    payslip_income: Optional[float],
    household_type: str,
    owns_private_property: bool,
    prev_hdb_loans: int,
    disposed_subsidised_last_30m: bool,
    employed: bool,
) -> dict:
    reasons = []

    # 1) Citizenship
    if str(row.get("Citizenship", "")).strip().lower() != "singapore citizen":
        reasons.append("At least one applicant must be a Singapore Citizen.")

    # 2) Household status (sim toggles)
    if owns_private_property:
        reasons.append("Owns private residential property within 30 months of application.")
    if disposed_subsidised_last_30m:
        reasons.append("Disposed of a subsidised flat within the last 30 months.")

    # 3) Income ceiling
    declared_income = float(row.get("Declared Income", 0) or 0)
    ceiling = INCOME_CEILING.get(household_type, 14000)
    if declared_income > ceiling:
        reasons.append(
            f"Declared income (S${declared_income:,.0f}) exceeds ceiling for {household_type} (S${ceiling:,.0f})."
        )

    # 4) Employment & payslip consistency
    if not employed:
        reasons.append("Applicant not employed / unable to provide proof of stable income.")
    if payslip_income is None:
        reasons.append("Payslip income could not be read. Please re-upload a clearer payslip.")
    else:
        if declared_income > 0:
            diff_ratio = abs(payslip_income - declared_income) / declared_income
            if diff_ratio > 0.10:
                reasons.append(
                    f"Payslip income (S${payslip_income:,.0f}) does not match declared income (S${declared_income:,.0f}) beyond 10%."
                )

    # 5) Loan history
    if prev_hdb_loans >= 2:
        reasons.append("Has taken two or more previous HDB housing loans.")

    decision = "ACCEPT" if not reasons else "REJECT"
    return {
        "decision": decision,
        "reasons": reasons,
        "declared_income": declared_income,
        "payslip_income": payslip_income,
        "ceiling": ceiling,
    }

# =========================
# Robust CSV loader (auto-create if missing)
# =========================
SAMPLE_ROWS = [
    ["Alice Tan","Singapore Citizen","S1234567A","Female","Single","1995-04-21",4800],
    ["Brandon Lim","Singapore PR","S7654321B","Male","Married","1988-11-03",10000],
    ["Cheryl Ng","Singapore Citizen","S2345678C","Female","Married","1992-06-14",13500],
    ["Dan Wong","Singapore Citizen","S3456789D","Male","Single","1998-02-10",6500],
    ["Evelyn Goh","Singapore Citizen","S4567890E","Female","Married","1985-09-27",7200],
    ["Farhan Ahmad","Singapore Citizen","S5678901F","Male","Married","1990-01-12",14500],
    ["Grace Lee","Singapore PR","S6789012G","Female","Single","1997-07-19",5500],
    ["Henry Koh","Singapore Citizen","S7890123H","Male","Single","1994-03-08",3000],
    ["Irene Chua","Singapore Citizen","S8901234I","Female","Married","1982-12-01",21000],
    ["Jason Ong","Singapore Citizen","S9012345J","Male","Married","1986-05-15",6800],
    ["Kelly Lim","Singapore Citizen","S0123456K","Female","Single","1999-08-22",2500],
    ["Leonard Ho","Singapore Citizen","S1122334L","Male","Married","1979-06-30",15500],
    ["Melissa Tan","Singapore Citizen","S2233445M","Female","Married","1991-11-11",9000],
    ["Nicholas Chan","Singapore PR","S3344556N","Male","Single","1993-10-20",4000],
    ["Olivia Wong","Singapore Citizen","S4455667O","Female","Single","1996-04-02",7100],
    ["Peter Lim","Singapore Citizen","S5566778P","Male","Married","1984-09-14",12000],
    ["Queenie Lau","Singapore Citizen","S6677889Q","Female","Married","1987-07-07",13400],
    ["Raymond Tan","Singapore Citizen","S7788990R","Male","Single","1992-05-21",2000],
    ["Sophia Ng","Singapore PR","S8899001S","Female","Single","1995-01-17",6500],
    ["Thomas Goh","Singapore Citizen","S9900112T","Male","Married","1989-03-25",8000],
]
COLUMNS = ["Full Name","Citizenship","IC","Sex","Marital Status","Date of Birth","Declared Income"]

@st.cache_data
def load_db() -> pd.DataFrame:
    base = Path(__file__).resolve().parent
    candidates = [base / "data" / "mock_db.csv", base / "mock_db.csv"]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    # auto-create if missing
    data_dir = base / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / "mock_db.csv"
    df = pd.DataFrame(SAMPLE_ROWS, columns=COLUMNS)
    df.to_csv(out_path, index=False)
    return df

# =========================
# Helpers
# =========================
IC_REGEX = re.compile(r"\b([STFG]\d{7}[A-Z])\b")

def mask_ic(ic: str) -> str:
    if not ic or len(ic) < 3:
        return ic
    return ic[:2] + "*****" + ic[-1:]

# ===== Improved JSON explanation with next steps & few-shot =====
def llm_explain_hosted_json(decision: str, reasons: List[str], profile: Dict) -> Dict:
    """
    Returns a compact, strictly-JSON explanation with suggested next steps.
    Schema:
      {
        "summary": str,              # 1–2 sentences, plain English, no emojis
        "tone": "accept"|"reject",   # must match decision
        "next_steps": [str, ...],    # 2–5 imperative steps
        "notes": str | null          # optional, 1 sentence max
      }
    """
    import re, json

    sys_msg = (
        "You are a helpful HDB assistant.\n"
        "Respond ONLY with a single valid JSON object. No markdown, no prefix/suffix text.\n"
        "Use this JSON schema exactly:\n"
        "{\n"
        '  "summary": "string (1-2 sentences, plain English, no emojis)",\n'
        '  "tone": "accept" | "reject",\n'
        '  "next_steps": ["string", ...],\n'
        '  "notes": "string or null"\n'
        "}\n"
        "Constraints:\n"
        "- Keep facts consistent with inputs.\n"
        '- If decision is ACCEPT, tone MUST be "accept"; if REJECT, tone MUST be "reject".\n'
        "- Be specific and practical; avoid legalese.\n"
        "- No advice that contradicts the reasons."
    )

    # Few-shot ACCEPT
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

    # Few-shot REJECT
    ex_reject_user = (
        "Decision: REJECT\n"
        "Reasons: Owns private residential property within 30 months of application., Payslip income (S$9,500) does not match declared income (S$7,000) beyond 10%.\n"
        "Profile: {\"Name\":\"Peter Lim\",\"Citizenship\":\"Singapore Citizen\",\"Marital Status\":\"Married\",\"Declared Income\":7000,\"Payslip Income\":9500,\"Household Type\":\"Family\"}\n"
        "Write the JSON."
    )
    ex_reject_assistant = {
        "summary": "You do not currently meet HDB loan criteria due to private property ownership within the last 30 months and a large mismatch between declared and payslip income.",
        "tone": "reject",
        "next_steps": [
            "Confirm the disposal date for any private property and wait until 30 months have passed",
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
        {"role": "assistant", "content": json.dumps(ex_accept_assistant, ensure_ascii=False)},
        {"role": "user", "content": ex_reject_user},
        {"role": "assistant", "content": json.dumps(ex_reject_assistant, ensure_ascii=False)},
        {"role": "user", "content": user_msg},
    ]

    raw = llama3_chat_safe(messages, max_tokens=220, temperature=0.2, top_p=0.9)

    if raw.startswith("(LLM error"):
        return {"summary": raw, "tone": decision.lower(), "next_steps": [], "notes": None}

    s = raw.strip()
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) > 1 else s

    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return {"summary": s[:180], "tone": decision.lower(), "next_steps": [], "notes": None}

    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {"summary": s[:180], "tone": decision.lower(), "next_steps": [], "notes": None}

    # Minimal schema guardrails
    obj.setdefault("summary", "")
    obj.setdefault("tone", decision.lower())
    obj.setdefault("next_steps", [])
    obj.setdefault("notes", None)

    # Force tone to match decision
    obj["tone"] = "accept" if decision.upper() == "ACCEPT" else "reject"

    # Normalize & trim next_steps to 2–5 items
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
                "Resolve the listed issues (e.g., property ownership window, document mismatches)",
                "Re-check eligibility after corrections"
            ]
        obj["next_steps"] = obj["next_steps"][:5]

    obj["summary"] = " ".join(str(obj["summary"]).split())[:400]
    if obj["notes"] is not None:
        obj["notes"] = " ".join(str(obj["notes"]).split())[:200]

    return obj

# =========================
# Streamlit UI (cleaned)
# =========================
st.set_page_config(page_title="🏠 HDB Loan Chatbot – Llama 3 8B (Hosted)", page_icon="🏠", layout="centered")
st.title("🏠 HDB Loan Chatbot – Llama 3 8B (Hosted)")

# ---- Sidebar: keep only Reset ----
st.sidebar.header("Actions")
if st.sidebar.button("🔄 Reset chat"):
    st.session_state.clear()
    st.rerun()

# ---- Defaults (no sidebar LLM controls) ----
GPT_TEMP = 0.2
GPT_TOP_P = 0.9
MAX_LEN = 220

# ---- Non-DB criteria (main area) ----
with st.expander("Simulator – Non-DB Criteria", expanded=False):
    household_type = st.selectbox("Household Type", ["Family", "Extended Family", "Single"], index=0)
    owns_private_property = st.checkbox("Owns Private Property (last 30 months)", value=False)
    prev_hdb_loans = st.number_input("Previous HDB Loans Taken", min_value=0, max_value=5, value=0, step=1)
    disposed_subsidised_last_30m = st.checkbox("Disposed subsidised flat within last 30 months", value=False)
    employed = st.checkbox("Applicant is employed (stable income)", value=True)

# ---- Load DB ----
try:
    df = load_db()
except Exception as e:
    st.error(f"Failed to load mock DB: {e}")
    st.stop()

# ---- Conversation state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

def chat(role: str, content):
    """Append only strings to history to avoid rendering Streamlit objects."""
    if not isinstance(content, str):
        try:
            content = json.dumps(content, ensure_ascii=False)
        except Exception:
            content = str(content)
    st.session_state.messages.append({"role": role, "content": content})

# ---- Greeting ----
if not st.session_state.messages:
    chat("assistant", "Hello! I’m your HDB assistant. Please enter your **IC** (e.g., S1234567A) to find your record.")

# ---- Render history ----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        content = m.get("content", "")
        if isinstance(content, str):
            st.markdown(content)
        elif isinstance(content, dict):
            st.json(content)
        else:
            st.markdown(str(content))

# ---- User input ----
user_text = st.chat_input("Type here… e.g., Check my eligibility. My IC is S1234567A")

if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.messages.append({"role": "user", "content": user_text})

    ic_match = IC_REGEX.search(user_text.upper())
    if not ic_match:
        chat("assistant", "Please provide your **IC** (e.g., S1234567A) so I can check your records.")
    else:
        ic = ic_match.group(1)
        hit = df[df["IC"].str.upper() == ic]
        if hit.empty:
            chat("assistant", f"I couldn’t find records for IC **{mask_ic(ic)}**. You may try another IC or ask an HDB officer for assistance.")
        else:
            row = hit.iloc[0]
            profile_md = (
                f"**Name:** {row['Full Name']}  \n"
                f"**Citizenship:** {row['Citizenship']}  \n"
                f"**Sex:** {row['Sex']}  \n"
                f"**Marital Status:** {row['Marital Status']}  \n"
                f"**Date of Birth:** {row['Date of Birth']}  \n"
                f"**Declared Income:** S${float(row['Declared Income']):,.0f}  \n\n"
                "Please upload your **latest payslip (PDF)** so I can verify your income."
            )
            chat("assistant", profile_md)

# ---- Upload area when prompted ----
needs_upload = any("upload" in str(m["content"]).lower() for m in st.session_state.messages if m["role"] == "assistant")
if needs_upload:
    uploaded = st.file_uploader(
        "Upload payslip (PDF, PNG, or JPEG)", 
        type=["pdf", "png", "jpg", "jpeg"]
    )
    if uploaded is not None:
        # Ensure GOOGLE_API_KEY is present before calling Gemini
        has_gemini_key = bool(os.environ.get("GOOGLE_API_KEY"))
        try:
            if not has_gemini_key:
                _ = st.secrets["GOOGLE_API_KEY"]  # type: ignore[index]
                has_gemini_key = True
        except Exception:
            pass

        if not has_gemini_key:
            chat("assistant", "Missing **GOOGLE_API_KEY**. Please set it in your environment or `.streamlit/secrets.toml` and rerun.")
        else:
            with st.spinner("Reading payslip…"):
                payslip_income = parse_payslip(uploaded)

            if payslip_income is None:
                chat("assistant", "Sorry, I couldn’t read a clear **Monthly Basic Salary** from your payslip. Please upload a clearer PDF.")
            else:
                chat("assistant", f"Thanks! I read a monthly basic salary of **S${payslip_income:,.0f}** from your payslip. Let me check your eligibility…")

                # find latest IC from chat
                ic_vals = [IC_REGEX.search(str(m["content"]).upper()).group(1)  # type: ignore[union-attr]
                           for m in st.session_state.messages
                           if m["role"] == "user" and IC_REGEX.search(str(m["content"]).upper())]
                row = None
                if ic_vals:
                    ic = ic_vals[-1]
                    hits = df[df["IC"].str.upper() == ic]
                    if not hits.empty:
                        row = hits.iloc[0]

                if row is None:
                    chat("assistant", "I lost track of your IC. Please resend it and re-upload your payslip.")
                else:
                    result = check_eligibility(
                        row,
                        payslip_income,
                        household_type,
                        owns_private_property,
                        prev_hdb_loans,
                        disposed_subsidised_last_30m,
                        employed,
                    )

                    profile = {
                        "Name": row["Full Name"],
                        "Citizenship": row["Citizenship"],
                        "Marital Status": row["Marital Status"],
                        "Declared Income": float(row["Declared Income"]),
                        "Payslip Income": payslip_income,
                        "Household Type": household_type,
                    }

                    # Decision card
                    if result["decision"] == "ACCEPT":
                        with st.chat_message("assistant"):
                            st.success(
                                "✅ You appear ELIGIBLE for an HDB loan based on this quick check.\n\n"
                                f"• Household type: **{household_type}** (ceiling **S${result['ceiling']:,.0f}**)\n"
                                f"• Declared income: **S${result['declared_income']:,.0f}**\n"
                                f"• Payslip income: **S${(result['payslip_income'] or 0):,.0f}**\n\n"
                                "This is a preliminary assessment. Please proceed to apply for an **HFE Letter** via the HDB Flat Portal for confirmation.",
                                icon="✅",
                            )
                        st.session_state.messages.append({"role": "assistant", "content": "ELIGIBLE summary shown"})
                    else:
                        reasons_md = "\n".join([f"- {r}" for r in result["reasons"]])
                        with st.chat_message("assistant"):
                            st.error(
                                "❌ Sorry, you do not appear to qualify based on this quick check.\n\n"
                                f"**Reasons**:\n{reasons_md}\n\n"
                                "You may revisit your application and **check with an HDB officer**.",
                                icon="❌",
                            )
                        st.session_state.messages.append({"role": "assistant", "content": "INELIGIBLE summary shown"})

                    # LLM explanation (deterministic JSON with next_steps)
                    with st.spinner("Generating explanation…"):
                        llm = llm_explain_hosted_json(result["decision"], result["reasons"], profile)

                    # Render explanation
                    with st.chat_message("assistant"):
                        if llm.get("summary"):
                            st.markdown(f"**Explanation:** {llm['summary']}")
                        steps = llm.get("next_steps") or []
                        if steps:
                            st.markdown("**What to do next:**")
                            for s in steps:
                                st.markdown(f"- {s}")
                        if llm.get("notes"):
                            st.caption(llm["notes"])

st.divider()
with st.expander("Debug", expanded=False):
    st.write(f"Active Google key index: {st.session_state.gapi_idx} / {max(1, len(GOOGLE_KEYS))}")
st.divider()
st.caption(
    "Demo only. Criteria simplified for illustration. Income parsing is heuristic. "
    "Do not upload real personal data; use sample payslips for demos."
)
