"""
🏠 HDB Loan Chatbot – Llama 3 8B (Hugging Face Hosted) + Gemini Payslip Parsing

Quick start:
  pip install --upgrade streamlit pandas PyPDF2 "huggingface_hub>=0.28.0" google-generativeai
  export HUGGINGFACE_API_KEY=hf_...
  export GOOGLE_API_KEY=AIza...
  streamlit run hdb_loan_chatbot_llama3_streamlit.py

Notes:
- Uses HF Chat Completions (task=conversational) for meta-llama/Meta-Llama-3-8B-Instruct
- Uses Gemini (explicit Client, no ADC) to parse payslip PDFs
- Requires fine-grained HF token & a Gemini API key
"""

import os
import io
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

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
# Hugging Face Inference API (Hosted, Chat Completions)
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
    max_tokens: int = 160,
    temperature: float = 0.7,
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


def llama3_chat_stream(
    messages: List[Dict[str, str]],
    max_tokens: int = 160,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Streaming generator of text deltas. Yields strings (delta tokens).
    """
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
            # be resilient to provider quirks
            pass

# =========================
# Payslip parsing (Gemini → returns float basic salary)
# =========================
import json
import re
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


@st.cache_data
def load_db() -> pd.DataFrame:
    df = pd.read_csv('data/mock_db.csv')
    return df

# =========================
# Helpers
# =========================
IC_REGEX = re.compile(r"\b([STFG]\d{7}[A-Z])\b")

def mask_ic(ic: str) -> str:
    if not ic or len(ic) < 3:
        return ic
    return ic[:2] + "*****" + ic[-1:]


def llm_explain_hosted_json(decision: str, reasons: List[str], profile: Dict) -> Dict:
    """
    Ask the model to return a compact JSON payload for simple rendering.
    Falls back to plain text if JSON parsing fails.
    """
    system = (
        "You are a helpful HDB assistant. Respond ONLY with a compact JSON object "
        'like {"summary":"...", "tone":"accept|reject"} and nothing else.'
    )
    user = (
        f"Decision: {decision}\n"
        f"Reasons: {', '.join(reasons) if reasons else 'None'}\n"
        f"Profile: {profile}\n"
        f'Write 1–2 sentences in "summary". Set "tone" to "accept" or "reject".'
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    raw = llama3_chat_safe(messages, max_tokens=160, temperature=0.3, top_p=0.9)
    import json
    if raw.startswith("(LLM error"):
        return {"summary": raw, "tone": decision.lower()}
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {"summary": raw.strip(), "tone": decision.lower()}

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

# ---- Default LLM params (no sidebar controls) ----
GPT_TEMP = 0.7
GPT_TOP_P = 0.9
MAX_LEN = 160
STREAM_OUTPUT = True

# ---- Non-DB criteria (moved out of sidebar) ----
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
            import json
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
    uploaded = st.file_uploader("Upload payslip (PDF only)", type=["pdf"])
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

                    # LLM explanation (non-streaming JSON for reliability)
                    with st.spinner("Generating explanation…"):
                        llm = llm_explain_hosted_json(result["decision"], result["reasons"], profile)

                    tone = (llm.get("tone") or result["decision"].lower()).strip()
                    summary = llm.get("summary", "")

                    # Optional natural-language elaboration (uses fixed defaults)
                    if STREAM_OUTPUT and isinstance(summary, str) and len(summary) < 40:
                        messages = [
                            {"role": "system", "content": "You are a polite HDB assistant. Keep under 120 words."},
                            {"role": "user", "content": f"Rewrite this more clearly (keep facts consistent): {summary}"},
                        ]
                        with st.chat_message("assistant"):
                            placeholder = st.empty()
                            acc = ""
                            for delta in llama3_chat_stream(messages, max_tokens=MAX_LEN, temperature=GPT_TEMP, top_p=GPT_TOP_P):
                                acc += delta
                                placeholder.markdown(acc)
                            chat("assistant", acc)
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(f"**Explanation:** {summary}")
                        chat("assistant", summary)



st.divider()
with st.expander("Debug", expanded=False):
    st.write(f"Active Google key index: {st.session_state.gapi_idx} / {max(1, len(GOOGLE_KEYS))}")
st.divider()
st.caption(
    "Demo only. Criteria simplified for illustration. Income parsing is heuristic. "
    "Do not upload real personal data; use sample payslips for demos."
)
