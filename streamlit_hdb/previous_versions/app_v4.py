"""
🏠 HDB Loan Chatbot – Llama 3 8B (Hugging Face Hosted) + Gemini Payslip Parsing
Dual-track chat: Always answers general HDB questions AND runs IC→payslip→eligibility flow when an IC is present.
Uploader only appears AFTER a valid IC is entered and found. Replies render immediately after each user turn.

Quick start:
  pip install --upgrade streamlit pandas PyPDF2 "huggingface_hub>=0.28.0" google-generativeai
  export HUGGINGFACE_API_KEY=hf_...
  export GOOGLE_API_KEYS="AIza...1,AIza...2"   # optional rotation (comma-separated)
  # or: export GOOGLE_API_KEY=AIza...          # single-key fallback
  streamlit run hdb_loan_chatbot_llama3_streamlit.py

Notes:
- Uses HF Chat Completions for meta-llama/Meta-Llama-3-8B-Instruct (hosted).
- Uses Gemini (no Client object) to parse payslips (PDF/PNG/JPG).
- Demo only—figures/policies change. Verify on official HDB portals.
"""

import os
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
        env_keys = os.environ.get("GOOGLE_API_KEYS", "")
        if env_keys:
            keys = [k.strip() for k in env_keys.split(",") if k.strip()]
    return keys

GOOGLE_KEYS = _load_google_keys()
if not GOOGLE_KEYS:
    st.warning("No GOOGLE_API_KEYS configured. Falling back to single GOOGLE_API_KEY (if set).")

if "gapi_idx" not in st.session_state:
    st.session_state.gapi_idx = 0  # round-robin pointer

def _current_google_key() -> Optional[str]:
    if GOOGLE_KEYS:
        return GOOGLE_KEYS[st.session_state.gapi_idx % len(GOOGLE_KEYS)]
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

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # accept model license on HF

@st.cache_resource(show_spinner=True)
def get_hf_client() -> InferenceClient:
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
    if not key:
        raise RuntimeError("Missing Google API key.")
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-2.5-flash")

def gemini_generate_with_rotation(parts: list, max_attempts: int = None):
    attempts = max_attempts or max(1, len(GOOGLE_KEYS))
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
            if any(t in msg for t in ["429", "quota", "rate", "exhaust"]):
                _advance_key()
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

    if str(row.get("Citizenship", "")).strip().lower() != "singapore citizen":
        reasons.append("At least one applicant must be a Singapore Citizen.")

    if owns_private_property:
        reasons.append("Owns private residential property within 30 months of application.")
    if disposed_subsidised_last_30m:
        reasons.append("Disposed of a subsidised flat within the last 30 months.")

    declared_income = float(row.get("Declared Income", 0) or 0)
    ceiling = INCOME_CEILING.get(household_type, 14000)
    if declared_income > ceiling:
        reasons.append(
            f"Declared income (S${declared_income:,.0f}) exceeds ceiling for {household_type} (S${ceiling:,.0f})."
        )

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
    df = pd.read_csv('data/mock_db.cvs')
    return df

# =========================
# Helpers & LLM JSON explanation
# =========================
IC_REGEX = re.compile(r"\b([STFG]\d{7}[A-Z])\b")

def mask_ic(ic: str) -> str:
    if not ic or len(ic) < 3:
        return ic
    return ic[:2] + "*****" + ic[-1:]

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
                "Resolve the listed issues (e.g., property ownership window, document mismatches)",
                "Re-check eligibility after corrections"
            ]
        obj["next_steps"] = obj["next_steps"][:5]
    obj["summary"] = " ".join(str(obj["summary"]).split())[:400]
    if obj["notes"] is not None:
        obj["notes"] = " ".join(str(obj["notes"]).split())[:200]
    return obj

# =========================
# NEW: General HDB Q&A helpers (always-on)
# =========================
QUESTION_HINTS = [
    "eligibility", "apply", "application", "hfe", "hfe letter", "loan",
    "income ceiling", "grants", "bto", "resale", "ec", "executive condominium",
    "cpf", "cpf oa", "tenure", "interest", "msr", "valuation", "lease",
    "hdb flat portal", "documents", "payslip", "noa", "notice of assessment",
    "age", "citizen", "pr", "private property", "cooling period",
    "mismatch", "appeal", "reject", "accept", "bank loan", "hdb loan"
]

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
    msgs.append({"role": "user", "content": user_text.strip()})
    out = llama3_chat_safe(msgs, max_tokens=300, temperature=0.2, top_p=0.9)
    if not out or out.startswith("(LLM error"):
        return "Here’s a quick overview. Criteria may change; please check the HDB Flat Portal for the latest details."
    s = " ".join(out.split())
    if "HDB Flat Portal" not in s and "HFE" not in s:
        s += " For the latest policy details, please check the HDB Flat Portal."
    return s

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="🏠 HDB Loan Chatbot – Llama 3 8B (Hosted)", page_icon="🏠", layout="centered")
st.title("🏠 HDB Loan Chatbot – Llama 3 8B (Hosted)")

# ---- Sidebar: Reset ----
st.sidebar.header("Actions")
if st.sidebar.button("🔄 Reset chat"):
    st.session_state.clear()
    st.rerun()

# ---- Non-DB criteria (main area)
with st.expander("Simulator – Non-DB Criteria", expanded=False):
    household_type = st.selectbox("Household Type", ["Family", "Extended Family", "Single"], index=0)
    st.session_state["household_type"] = household_type  # store for Q&A context
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

# NEW: control uploader visibility + track which IC we're processing
if "awaiting_payslip" not in st.session_state:
    st.session_state.awaiting_payslip = False
if "current_ic" not in st.session_state:
    st.session_state.current_ic = None

def log(role: str, content):
    """Append to message history only (used for initial greeting)."""
    if not isinstance(content, str):
        try:
            content = json.dumps(content, ensure_ascii=False)
        except Exception:
            content = str(content)
    st.session_state.messages.append({"role": role, "content": content})

def say(content: str):
    """Render assistant message NOW and log it to history."""
    with st.chat_message("assistant"):
        st.markdown(content)
    st.session_state.messages.append({"role": "assistant", "content": content})

# ---- Greeting ----
if not st.session_state.messages:
    log("assistant",
        "Hello! I’m your HDB assistant. Ask any **general HDB loan questions** and/or enter your **IC** "
        "(e.g., S1234567A) to look up a demo record and simulate eligibility. "
        "You can upload a payslip later (PDF/PNG/JPG).")

# ---- Render history (for past messages only)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        content = m.get("content", "")
        if isinstance(content, str):
            st.markdown(content)
        elif isinstance(content, dict):
            st.json(content)
        else:
            st.markdown(str(content))

# ---- User input (dual-track routing)
user_text = st.chat_input("Type here… ask anything about HDB loans, or include your IC to check eligibility")

if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.messages.append({"role": "user", "content": user_text})

    # 1) Always try to answer general questions if the text looks like one
    did_general = False
    if looks_like_question(user_text):
        profile_hint = None
        try:
            prev_ics = [
                IC_REGEX.search(str(m["content"]).upper()).group(1)  # type: ignore[union-attr]
                for m in st.session_state.messages
                if m["role"] == "user" and IC_REGEX.search(str(m["content"]).upper())
            ]
            if prev_ics:
                last_ic = prev_ics[-1]
                prev_hit = df[df["IC"].str.upper() == last_ic]
                if not prev_hit.empty:
                    pr = prev_hit.iloc[0]
                    profile_hint = {
                        "Citizenship": pr.get("Citizenship", ""),
                        "Household Type (selected)": st.session_state.get("household_type", None) or "Not set",
                        "Declared Income": float(pr.get("Declared Income", 0) or 0),
                    }
        except Exception:
            profile_hint = None

        ans = hdb_general_answer(user_text, profile_hint)
        say(ans)
        did_general = True

    # 2) Also attempt the IC-based original flow in the same turn (if any IC present)
    ic_match = IC_REGEX.search(user_text.upper())
    if ic_match:
        ic = ic_match.group(1)
        hit = df[df["IC"].str.upper() == ic]
        if hit.empty:
            say(f"I couldn’t find records for IC **{mask_ic(ic)}**. "
                f"You can try another IC or continue asking HDB questions.")
            st.session_state.awaiting_payslip = False
            st.session_state.current_ic = None
        else:
            row = hit.iloc[0]
            profile_md = (
                f"**Name:** {row['Full Name']}  \n"
                f"**Citizenship:** {row['Citizenship']}  \n"
                f"**Sex:** {row['Sex']}  \n"
                f"**Marital Status:** {row['Marital Status']}  \n"
                f"**Date of Birth:** {row['Date of Birth']}  \n"
                f"**Declared Income:** S${float(row['Declared Income']):,.0f}  \n\n"
                "Please upload your **latest payslip (PDF/PNG/JPG)** so I can verify your income."
            )
            say(profile_md)
            st.session_state.awaiting_payslip = True
            st.session_state.current_ic = ic
    else:
        if not did_general:
            say("You can **enter your IC** (e.g., S1234567A) to check your record and simulate eligibility, "
                "or ask any HDB loan question.")

# ---- Upload area: ONLY when a valid IC was provided this session
if st.session_state.get("awaiting_payslip", False):
    uploaded = st.file_uploader(
        "Upload payslip (PDF, PNG, or JPEG)",
        type=["pdf", "png", "jpg", "jpeg"]
    )
    if uploaded is not None:
        has_gemini_key = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEYS"))
        try:
            if not has_gemini_key:
                _ = st.secrets["GOOGLE_API_KEY"]  # type: ignore[index]
                has_gemini_key = True
        except Exception:
            try:
                _ = st.secrets["GOOGLE_API_KEYS"]  # type: ignore[index]
                has_gemini_key = True
            except Exception:
                pass

        if not has_gemini_key:
            say("Missing **GOOGLE_API_KEY/GOOGLE_API_KEYS**. Please set it in your environment or `.streamlit/secrets.toml` and rerun.")
        else:
            with st.spinner("Reading payslip…"):
                payslip_income = parse_payslip(uploaded)

            if payslip_income is None:
                say("Sorry, I couldn’t read a clear **Monthly Basic Salary** from your payslip. "
                    "Please upload a clearer file (PDF/PNG/JPG).")
                # Keep awaiting_payslip = True so user can try again
            else:
                say(f"Thanks! I read a monthly basic salary of **S${payslip_income:,.0f}** from your payslip. "
                    f"Let me check your eligibility…")

                ic_to_use = st.session_state.get("current_ic")
                if not ic_to_use:
                    ic_vals = [IC_REGEX.search(str(m["content"]).upper()).group(1)  # type: ignore[union-attr]
                               for m in st.session_state.messages
                               if m["role"] == "user" and IC_REGEX.search(str(m["content"]).upper())]
                    if ic_vals:
                        ic_to_use = ic_vals[-1]

                row = None
                if ic_to_use:
                    hits = df[df["IC"].str.upper() == ic_to_use.upper()]
                    if not hits.empty:
                        row = hits.iloc[0]

                if row is None:
                    say("I lost track of your IC. Please resend it and re-upload your payslip.")
                    st.session_state.awaiting_payslip = False
                    st.session_state.current_ic = None
                else:
                    result = check_eligibility(
                        row,
                        payslip_income,
                        st.session_state.get("household_type", "Family"),
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
                        "Household Type": st.session_state.get("household_type", "Family"),
                    }

                    if result["decision"] == "ACCEPT":
                        with st.chat_message("assistant"):
                            st.success(
                                "✅ You appear ELIGIBLE for an HDB loan based on this quick check.\n\n"
                                f"• Household type: **{st.session_state.get('household_type', 'Family')}** (ceiling **S${result['ceiling']:,.0f}**)\n"
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

                    with st.spinner("Generating explanation…"):
                        llm = llm_explain_hosted_json(result["decision"], result["reasons"], profile)

                    # Render explanation immediately
                    expl = []
                    if llm.get("summary"):
                        expl.append(f"**Explanation:** {llm['summary']}")
                    steps = llm.get("next_steps") or []
                    if steps:
                        expl.append("**What to do next:**")
                        expl.extend([f"- {s}" for s in steps])
                    if llm.get("notes"):
                        expl.append(f"*{llm['notes']}*")
                    say("\n".join(expl) if expl else "Explanation generated.")

                    # Hide uploader after eligibility shown
                    st.session_state.awaiting_payslip = False
                    st.session_state.current_ic = None

# ---- Footer
st.divider()
with st.expander("Debug", expanded=False):
    st.write(f"Active Google key index: {st.session_state.gapi_idx} / {max(1, len(GOOGLE_KEYS))}")
    st.write(f"Awaiting payslip: {st.session_state.get('awaiting_payslip', False)} | Current IC: {st.session_state.get('current_ic')}")
st.divider()
st.caption(
    "Demo only. Criteria simplified for illustration. Policies update over time—verify on HDB portals. "
    "Do not upload real personal data; use sample payslips for demos."
)
