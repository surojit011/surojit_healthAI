"""
# Developed by: Surojit
# Project: AI Disease Prediction System
# Hackathon: AI in Healthcare & Life Sciences
app.py  –  AI Disease Prediction System (v2)
Features: AI Chat, Symptom Checkboxes, Nearby Hospital Finder
Run with:  python -m streamlit run app.py
"""

import pickle, os
import numpy as np
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #0d6efd 0%, #0a58ca 100%);
        border-radius: 16px; padding: 2rem 2.5rem; color: white;
        margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(13,110,253,0.3);
    }
    .hero h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .hero p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem; }

    .chat-wrap {
        background: #f8fafc; border-radius: 12px; padding: 1rem;
        min-height: 300px; max-height: 420px; overflow-y: auto;
        border: 1px solid #e2e8f0; margin-bottom: 1rem;
    }
    .bubble-user {
        background: #0d6efd; color: white; border-radius: 18px 18px 4px 18px;
        padding: 0.7rem 1.1rem; margin: 0.4rem 0 0.4rem 20%;
        font-size: 0.95rem; display: block; clear: both;
    }
    .bubble-ai {
        background: white; color: #1a1a2e; border-radius: 18px 18px 18px 4px;
        padding: 0.7rem 1.1rem; margin: 0.4rem 20% 0.4rem 0;
        font-size: 0.95rem; display: block; clear: both;
        border-left: 3px solid #0d6efd;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .result-box {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border-left: 5px solid #10b981; border-radius: 10px;
        padding: 1.5rem; margin-top: 1rem;
    }
    .result-box .disease { font-size: 1.6rem; font-weight: 800; color: #065f46; }
    .result-box .conf    { font-size: 1rem; color: #047857; margin-top: 0.3rem; }

    .warning-box {
        background: #fff3cd; border-left: 5px solid #ffc107;
        border-radius: 10px; padding: 1rem 1.5rem; margin-top: 1rem;
        font-size: 0.88rem; color: #664d03;
    }

    .chip-grid { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.6rem 0; }
    .chip {
        background: #e0f0ff; color: #0d6efd; border-radius: 20px;
        padding: 0.3rem 0.8rem; font-size: 0.82rem; font-weight: 500;
    }

    .footer { text-align:center; color:#aaa; font-size:0.8rem; margin-top:2rem; }

    .stButton > button {
        background: linear-gradient(135deg, #0d6efd, #0a58ca);
        color: white; border: none; border-radius: 10px;
        padding: 0.55rem 1.5rem; font-size: 0.95rem; font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(BASE_DIR, "symptom_cols.pkl"), "rb") as f:
        symptom_cols = pickle.load(f)
    return model, le, symptom_cols

model, le, SYMPTOM_COLS = load_artifacts()

# ── Data ──────────────────────────────────────────────────────────────────────
SYMPTOM_LABELS = {
    "fever":"🌡️ Fever","cough":"😷 Cough","headache":"🤕 Headache",
    "fatigue":"😴 Fatigue","nausea":"🤢 Nausea","chest_pain":"💔 Chest Pain",
    "shortness_of_breath":"😮‍💨 Shortness of Breath","sore_throat":"🗣️ Sore Throat",
    "runny_nose":"🤧 Runny Nose","body_ache":"🦴 Body Ache","vomiting":"🤮 Vomiting",
    "diarrhea":"🚽 Diarrhea","rash":"🔴 Skin Rash","joint_pain":"🦵 Joint Pain",
    "dizziness":"💫 Dizziness","loss_of_appetite":"🍽️ Loss of Appetite",
    "sweating":"💦 Sweating","chills":"🥶 Chills",
}

KEYWORD_MAP = {
    "fever":"fever","temperature":"fever","high temp":"fever",
    "cough":"cough","coughing":"cough","coughs":"cough",
    "headache":"headache","head pain":"headache","head ache":"headache","head hurts":"headache",
    "tired":"fatigue","fatigue":"fatigue","weakness":"fatigue","weak":"fatigue","exhausted":"fatigue","tiredness":"fatigue",
    "nausea":"nausea","nauseated":"nausea","feel sick":"nausea","feeling sick":"nausea",
    "chest pain":"chest_pain","chest ache":"chest_pain","chest hurts":"chest_pain","chest hurt":"chest_pain",
    "breathless":"shortness_of_breath","breathing problem":"shortness_of_breath","cant breathe":"shortness_of_breath",
    "shortness of breath":"shortness_of_breath","short of breath":"shortness_of_breath",
    "sore throat":"sore_throat","throat pain":"sore_throat","throat hurts":"sore_throat","throat":"sore_throat",
    "runny nose":"runny_nose","running nose":"runny_nose","nose running":"runny_nose",
    "body ache":"body_ache","body pain":"body_ache","muscle pain":"body_ache","muscle ache":"body_ache","body hurts":"body_ache",
    "vomit":"vomiting","vomiting":"vomiting","throw up":"vomiting","threw up":"vomiting","puking":"vomiting",
    "diarrhea":"diarrhea","loose motion":"diarrhea","stomach upset":"diarrhea","loose stool":"diarrhea",
    "rash":"rash","skin rash":"rash","skin irritation":"rash","itching":"rash","itchy":"rash",
    "joint pain":"joint_pain","joint ache":"joint_pain","joints hurt":"joint_pain","joints pain":"joint_pain",
    "dizzy":"dizziness","dizziness":"dizziness","vertigo":"dizziness","spinning head":"dizziness",
    "no appetite":"loss_of_appetite","loss of appetite":"loss_of_appetite","not eating":"loss_of_appetite","cant eat":"loss_of_appetite",
    "sweating":"sweating","sweat":"sweating","night sweat":"sweating","excessive sweat":"sweating",
    "chills":"chills","shivering":"chills","shiver":"chills","feeling cold":"chills",
}

DISEASE_INFO = {
    "Flu":             ("Influenza (Flu)",  "Rest, hydration, antiviral medication if needed."),
    "Common Cold":     ("Common Cold",      "Rest, fluids, OTC cold remedies."),
    "COVID-19":        ("COVID-19",         "Isolate, consult a doctor, monitor oxygen levels."),
    "Pneumonia":       ("Pneumonia",        "Antibiotics (if bacterial), rest, medical supervision."),
    "Malaria":         ("Malaria",          "Antimalarial drugs — seek medical care immediately."),
    "Dengue":          ("Dengue Fever",     "Supportive care, hydration, avoid NSAIDs — see a doctor."),
    "Typhoid":         ("Typhoid Fever",    "Antibiotics and medical supervision required."),
    "Gastroenteritis": ("Gastroenteritis",  "Oral rehydration, bland diet, rest."),
    "Migraine":        ("Migraine",         "Pain relief, rest in a dark quiet room, preventive meds."),
    "Asthma":          ("Asthma",           "Bronchodilator inhaler, avoid triggers, consult a doctor."),
    "Heart Disease":   ("Heart Disease",    "Urgent medical evaluation — do not ignore chest pain."),
    "Anemia":          ("Anemia",           "Iron/B12 supplements, dietary changes, treat underlying cause."),
    "Chickenpox":      ("Chickenpox",       "Antihistamines, calamine lotion, rest, and isolation."),
    "Measles":         ("Measles",          "Supportive care, Vitamin A supplements, medical supervision."),
    "Tuberculosis":    ("Tuberculosis",     "Long-term antibiotic regimen — see a doctor immediately."),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict_disease(chosen):
    x = np.array([[1 if s in chosen else 0 for s in SYMPTOM_COLS]])
    pred_idx   = model.predict(x)[0]
    proba      = model.predict_proba(x)[0]
    confidence = round(proba[pred_idx] * 100, 1)
    disease    = le.inverse_transform([pred_idx])[0]
    top3_idx   = np.argsort(proba)[::-1][:3]
    top3       = [(le.inverse_transform([i])[0], round(proba[i]*100,1)) for i in top3_idx if proba[i]>0]
    return disease, confidence, top3

def parse_symptoms(text):
    text_lower = text.lower()
    found = set()
    for kw in sorted(KEYWORD_MAP.keys(), key=len, reverse=True):
        if kw in text_lower:
            found.add(KEYWORD_MAP[kw])
    return list(found)

def render_result(disease, confidence, top3, chosen):
    full_name, advice = DISEASE_INFO.get(disease, (disease, "Please consult a healthcare professional."))
    chip_html = "".join(f'<span class="chip">{SYMPTOM_LABELS.get(s,s)}</span>' for s in chosen)
    st.markdown(f'<div class="chip-grid">{chip_html}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-box">
        <div class="disease">🩺 {full_name}</div>
        <div class="conf">Confidence: {confidence}%</div>
        <hr style="border-color:#6ee7b7;margin:0.8rem 0;">
        <b>💊 Suggested Action:</b> {advice}
    </div>
    <div class="warning-box">
        ⚕️ <b>Disclaimer:</b> This AI tool is for educational purposes only.
        Always consult a qualified medical professional for proper diagnosis.
    </div>
    """, unsafe_allow_html=True)
    if len(top3) > 1:
        st.markdown("#### 🔄 Other Possible Conditions")
        for name, prob in top3[1:]:
            st.progress(int(prob), text=f"{name} — {prob}%")

# ── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "ai", "text": "👋 Hello! I'm your AI Health Assistant. Tell me how you're feeling — describe your symptoms in your own words and I'll help analyze what might be going on!"}
    ]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏥 AI Health Assistant</h1>
    <p>Chat with AI · Select symptoms · Find nearby hospitals — all in one place.</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat with AI", "☑️ Select Symptoms", "🗺️ Find Nearby Hospitals"])

# ═══════════════════════════════════════════════
# TAB 1 — AI CHAT
# ═══════════════════════════════════════════════
with tab1:
    st.markdown("### 💬 Tell the AI how you're feeling")
    st.caption("Type naturally — e.g. *'I have fever and headache since yesterday, feeling very weak'*")

    # Display chat history
    chat_html = '<div class="chat-wrap">'
    for msg in st.session_state.chat_history:
        css = "bubble-user" if msg["role"] == "user" else "bubble-ai"
        chat_html += f'<div class="{css}">{msg["text"]}</div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input row
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_input = st.text_input(
            "", placeholder="e.g. I have fever, body ache and chills since 2 days...",
            label_visibility="collapsed", key="chat_input"
        )
    with col_btn:
        send = st.button("Send ➤")

    if send and user_input.strip():
        user_text = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "text": user_text})

        found = parse_symptoms(user_text)

        if len(found) == 0:
            reply = (
                "🤔 I couldn't identify specific symptoms from that. "
                "Try mentioning things like: fever, cough, headache, body ache, nausea, rash, dizziness, etc."
            )
        elif len(found) == 1:
            sym_name = SYMPTOM_LABELS.get(found[0], found[0])
            reply = (
                f"I noticed you mentioned **{sym_name}**. "
                "Could you tell me about any other symptoms? The more you share, the more accurate my prediction will be!"
            )
        else:
            disease, confidence, top3 = predict_disease(found)
            full_name, advice = DISEASE_INFO.get(disease, (disease, "Please consult a doctor."))
            sym_list = ", ".join([SYMPTOM_LABELS.get(s, s) for s in found])
            others = ""
            if len(top3) > 1:
                other_names = " | ".join([f"{n} ({p}%)" for n, p in top3[1:]])
                others = f"\n\n🔄 **Other possibilities:** {other_names}"
            reply = (
                f"I detected these symptoms: **{sym_list}**\n\n"
                f"🩺 **Most likely condition: {full_name}** (Confidence: {confidence}%)\n\n"
                f"💊 **What to do:** {advice}"
                f"{others}\n\n"
                f"⚠️ *This is AI-based — always consult a real doctor for proper diagnosis.*"
            )

        st.session_state.chat_history.append({"role": "ai", "text": reply})
        st.rerun()

    col_clear, col_tip = st.columns([1, 3])
    with col_clear:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = [
                {"role": "ai", "text": "👋 Hello! I'm your AI Health Assistant. Tell me how you're feeling!"}
            ]
            st.rerun()
    with col_tip:
        st.caption("💡 Tip: Mention at least 2-3 symptoms for best results")

# ═══════════════════════════════════════════════
# TAB 2 — CHECKBOX SYMPTOMS
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("### ☑️ Select All Symptoms You Have")
    st.caption("Tick everything that applies, then click Predict.")

    cols = st.columns(2)
    selected = {}
    for i, sym in enumerate(SYMPTOM_COLS):
        label = SYMPTOM_LABELS.get(sym, sym.replace("_"," ").title())
        selected[sym] = cols[i % 2].checkbox(label, key=f"cb_{sym}")

    if st.button("🔍 Predict Disease"):
        chosen = [s for s, v in selected.items() if v]
        if len(chosen) < 2:
            st.warning("⚠️ Please select **at least 2 symptoms** for a meaningful prediction.")
        else:
            disease, confidence, top3 = predict_disease(chosen)
            render_result(disease, confidence, top3, chosen)

# ═══════════════════════════════════════════════
# TAB 3 — NEARBY HOSPITALS
# ═══════════════════════════════════════════════
with tab3:
    st.markdown("### 🗺️ Find Hospitals Near You")
    st.caption("Enter your city or area name to see hospitals on the map.")

    location_input = st.text_input(
        "📍 Your Location",
        placeholder="e.g. Kolkata, Bhatpara, Mumbai, Delhi..."
    )

    if st.button("🔍 Find Hospitals"):
        if not location_input.strip():
            st.warning("Please enter your city or area name first.")
        else:
            loc = location_input.strip()
            loc_encoded = loc.replace(" ", "+")
            maps_url = f"https://www.google.com/maps/search/hospitals+near+{loc_encoded}"

            st.success(f"✅ Showing hospitals near **{loc}**")

            st.markdown(f"""
            <div style="border-radius:12px; overflow:hidden; box-shadow:0 2px 16px rgba(0,0,0,0.12); margin-top:1rem;">
                <iframe
                    width="100%"
                    height="500"
                    frameborder="0"
                    style="border:0; display:block;"
                    src="https://maps.google.com/maps?q=hospitals+near+{loc_encoded}&output=embed"
                    allowfullscreen>
                </iframe>
            </div>
            <div style="text-align:center; margin-top:1rem;">
                <a href="{maps_url}" target="_blank"
                   style="background:#0d6efd; color:white; padding:0.6rem 1.8rem;
                          border-radius:10px; text-decoration:none; font-weight:600; font-size:0.95rem;">
                    🗺️ Open Full Map in Google Maps
                </a>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box" style="margin-top:2rem;">
        🚨 <b>Emergency?</b> Call <b>108</b> (Ambulance) or <b>112</b> (National Emergency) immediately!
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 AI Health Assistant")
    st.markdown("""
    **3 ways to use this app:**

    💬 **Chat** — Type symptoms in your own words

    ☑️ **Select** — Tick from a symptom list

    🗺️ **Hospital** — Find hospitals near you

    ---
    **15 Diseases Covered**
    Flu · COVID-19 · Dengue · Malaria · Typhoid · Pneumonia · Tuberculosis · Chickenpox · Measles · Asthma · Heart Disease · Anemia · Migraine · Gastroenteritis · Common Cold

    ---
    🚨 **Emergency Numbers (India)**
    - 🚑 Ambulance: **108**
    - 🚔 Police: **100**
    - 🔥 Fire: **101**
    - 📞 National: **112**

    ---
    ⚠️ *For educational use only.*
    *Not a substitute for medical advice.*
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">AI Health Assistant · Built with ❤️ using Python & Streamlit · Hackathon 2025</div>',
    unsafe_allow_html=True
)
