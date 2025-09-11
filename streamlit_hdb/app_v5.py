# app.py
import json
import streamlit as st
import pandas as pd

from hdb_backend import (
    IC_REGEX, looks_like_question, mask_ic,
    GOOGLE_KEYS, init_key_index, has_any_google_key,
    load_db, parse_payslip, hdb_general_answer,
    salary_only_decision,
    llm_explain_hosted_json,   # ← add this
)


# ---- Minimal render helpers
def log(role: str, content):
    if not isinstance(content, str):
        try:
            content = json.dumps(content, ensure_ascii=False)
        except Exception:
            content = str(content)
    st.session_state.messages.append({"role": role, "content": content})

def say(content: str):
    with st.chat_message("assistant"):
        st.markdown(content)
    st.session_state.messages.append({"role": "assistant", "content": content})

def md_escape(s: str) -> str:
    # escape a few common Markdown troublemakers
    return (
        s.replace("\\", "\\\\")
         .replace("_", r"\_")
         .replace("*", r"\*")
         .replace("`", r"\`")
    )

st.set_page_config(page_title="🏠 HDB HabiChat", page_icon="🏠", layout="centered")
st.title("🏠 HDB HabiChat")

# Key rotation state
init_key_index()
if not GOOGLE_KEYS:
    st.warning("No GOOGLE_API_KEYS configured. Falling back to single GOOGLE_API_KEY (if set).")

# ---- Sidebar: Reset ----
st.sidebar.header("Actions")
if st.sidebar.button("🔄 Reset chat"):
    st.session_state.clear()
    st.rerun()

# ---- Load DB ----
try:
    df = load_db()
except Exception as e:
    st.error(f"Failed to load mock DB: {e}")
    st.stop()

# ---- Conversation state ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_payslip" not in st.session_state:
    st.session_state.awaiting_payslip = False
if "current_ic" not in st.session_state:
    st.session_state.current_ic = None

# ---- Minimal render helpers
def log(role: str, content):
    if not isinstance(content, str):
        try:
            content = json.dumps(content, ensure_ascii=False)
        except Exception:
            content = str(content)
    st.session_state.messages.append({"role": role, "content": content})

def say(content: str):
    with st.chat_message("assistant"):
        st.markdown(content)
    st.session_state.messages.append({"role": "assistant", "content": content})

# ---- Greeting ----
if not st.session_state.messages:
    log(
        "assistant",
        "Welcome, Officer. I’m here to help with loan validation.\n\n"
        "Start by entering an **applicant’s IC (e.g., S1234567A)** to pull their record. "
        "Once retrieved, you can upload a **payslip (PDF/PNG/JPG)**, and I’ll process it to "
        "confirm income accuracy against the loan application details, giving you a quick "
        "**approve/reject suggestion.**"
    )

# ---- Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        content = m.get("content", "")
        if isinstance(content, str):
            st.markdown(content)
        elif isinstance(content, dict):
            st.json(content)
        else:
            st.markdown(str(content))

# ---- Chat input
user_text = st.chat_input("Type here… ask anything about HDB loans, or include your IC to fetch your record")

if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.messages.append({"role": "user", "content": user_text})

    # 1) General Q&A
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
                        "Declared Income": float(pr.get("Declared Income", 0) or 0),
                    }
        except Exception:
            profile_hint = None

        ans = hdb_general_answer(user_text, profile_hint)
        say(ans)
        did_general = True

    # 2) IC lookup
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
                "Please upload the **latest payslip (PDF/PNG/JPG)** and I’ll extract the **monthly basic salary** "
                "and provide a quick **salary-match decision (accept/reject)**."
            )
            say(profile_md)
            st.session_state.awaiting_payslip = True
            st.session_state.current_ic = ic
    else:
        if not did_general:
            say("You can **enter your IC** (e.g., S1234567A) to fetch a demo record, "
                "or ask any HDB loan question.")

# ---- Upload area (only after valid IC)
if st.session_state.get("awaiting_payslip", False):
    uploaded = st.file_uploader(
        "Upload payslip (PDF, PNG, or JPEG)",
        type=["pdf", "png", "jpg", "jpeg"]
    )
    if uploaded is not None:
        if not has_any_google_key():
            say("Missing **GOOGLE_API_KEY/GOOGLE_API_KEYS**. Please set it in your environment or `.streamlit/secrets.toml` and rerun.")
        else:
            with st.spinner("Reading payslip…"):
                payslip_income = parse_payslip(uploaded)

            ic_to_use = st.session_state.get("current_ic")
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
                if payslip_income is None:
                    # immediate reject
                    with st.chat_message("assistant"):
                        st.error(
                            "❌ Sorry, I couldn’t read a clear **Monthly Basic Salary** from your payslip. "
                            "Please upload a clearer file (PDF/PNG/JPG).",
                            icon="❌",
                        )
                        # Unreadable payslip branch
                        profile = {
                            "Name": row["Full Name"],
                            "Citizenship": row["Citizenship"],
                            "Declared Income": float(row.get("Declared Income", 0) or 0),
                            "Payslip Income": None,
                        }
                        with st.spinner("Generating explanation…"):
                            llm = llm_explain_hosted_json(
                                "REJECT",
                                ["Payslip income could not be read. Please re-upload a clearer payslip."],
                                profile,
                            )
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

                    st.session_state.messages.append({"role": "assistant", "content": "INELIGIBLE: unreadable payslip"})
                else:
                    say(f"I read a monthly basic salary of **S${payslip_income:,.0f}**. Evaluating salary match…")

                    result = salary_only_decision(row, payslip_income)
                    # After we compute `result = salary_only_decision(row, payslip_income)`

                    # Show ACCEPT/REJECT summary first
                    if result["decision"] == "ACCEPT":
                        with st.chat_message("assistant"):
                            st.success(
                                (
                                    "✅ **ACCEPT** (salary matches declared income within ±10%).\n\n"
                                    "- Declared income: **S${:,.0f}**\n"
                                    "- Payslip income: **S${:,.0f}**\n\n"
                                    "This is a basic salary-match check only. For official assessment, use the **HDB Flat Portal**."
                                ).format(result["declared_income"], result["payslip_income"]),
                                icon="✅",
                            )

                        st.session_state.messages.append({"role": "assistant", "content": "ACCEPT summary shown"})
                    else:
                        # Escape reasons to avoid Markdown italics / merging
                        reasons = [md_escape(r) for r in result.get("reasons", [])]
                        reasons_md = "\n".join([f"- {r}" for r in reasons]) or "- Not specified"
                    
                        with st.chat_message("assistant"):
                            st.error(
                                "❌ **REJECT** (based on salary-match rules).\n\n"
                                f"{reasons_md}\n\n"
                                "This is a basic salary-match check only. For official assessment, use the **HDB Flat Portal**.",
                                icon="❌",
                            )
                        st.session_state.messages.append({"role": "assistant", "content": "REJECT summary shown"})


                    # Build a compact profile for the explainer
                    profile = {
                        "Name": row["Full Name"],
                        "Citizenship": row["Citizenship"],
                        "Declared Income": float(row["Declared Income"]),
                        "Payslip Income": float(result.get("payslip_income") or (payslip_income or 0)),
                    }

                    # Generate LLM explanation + action steps
                    with st.spinner("Generating explanation…"):
                        llm = llm_explain_hosted_json(result["decision"], result.get("reasons", []), profile)

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

                    # Hide uploader after showing outcome + explanation
                    st.session_state.awaiting_payslip = False
                    st.session_state.current_ic = None

                    
# ---- Footer / Debug
st.divider()
st.caption(
    "Demo only. Decision here is based solely on salary match vs declared income (±10%). "
    "Policies change over time—verify on official HDB portals."
)
