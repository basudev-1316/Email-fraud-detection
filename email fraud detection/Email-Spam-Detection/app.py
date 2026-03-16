import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from pathlib import Path

nltk.download("stopwords", quiet=True)

# UI config
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="centered")
css = """ 
<style>
/* page background */
.stApp {
  background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%);
  color: #0f1724;
}
/* card */
.card {
  background: white;
  padding: 1rem;
  border-radius: 12px;
  box-shadow: 0 6px 20px rgba(16,24,40,0.08);
}
/* header */
.title {
  font-weight: 700;
  font-size: 28px;
  color: #0b2447;
}
.small-muted {
  color: #44516b;
  font-size: 13px;
}
.readout {
  background: linear-gradient(90deg,#7dd3fc,#60a5fa);
  color: white;
  padding: 0.6rem;
  border-radius: 10px;
  text-align:center;
  font-weight:700;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ensure nltk resources
for pkg in ("punkt", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

ps = PorterStemmer()

def transform_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # use regex tokenization to avoid depending on NLTK punkt/punkt_tab
    tokens = re.findall(r"\w+", text)
    words = [w for w in tokens if w.isalnum()]
    words = [w for w in words if w not in stopwords.words("english")]
    stems = [ps.stem(w) for w in words]
    return " ".join(stems)

# cached loader
@st.cache_resource
def load_artifacts(vpath="vectorizer.pkl", mpath="model.pkl"):
    base = Path(__file__).parent
    vec_file = base / vpath
    mod_file = base / mpath
    if not vec_file.exists() or not mod_file.exists():
        raise FileNotFoundError(f"Missing pickles. Expected: {vec_file}, {mod_file}")
    with open(vec_file, "rb") as vf:
        tfidf = pickle.load(vf)
    with open(mod_file, "rb") as mf:
        model = pickle.load(mf)
    return tfidf, model

# Sidebar
with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>📧 Email Spam Classifier</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Preprocessing: tokenization • stopword removal • stemming</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("## Examples")
    examples = {
        "Phishing / Spam (example)": "Congratulations! You have won a $1000 gift card. Click the link to claim your prize.",
        "Promotional (likely spam)": "Limited time offer — 70% off on all items. Visit https://example.com/deals to claim your discount.",
        "Personal (ham)": "Hey, are we still meeting tomorrow at 10am? Let me know what time works for you.",
        "Work (ham)": "Please find attached the project update and the meeting notes from today. Let's discuss in the 3pm standup."
    }
    choice = st.selectbox("Choose a sample message", ["-- none --"] + list(examples.keys()), index=0)
    # Always set the session value so switching back to "-- none --" clears the text area
    if choice == "-- none --":
        st.session_state["example_text"] = ""
    else:
        st.session_state["example_text"] = examples[choice]
    st.markdown("---")
    st.markdown("Model artifacts must be in this folder: <br> Email-Spam-Detection/vectorizer.pkl <br> Email-Spam-Detection/model.pkl", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Main
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'>"
            "<div><h2 style='margin:0'>📨 Predict Email Spam</h2><div class='small-muted'>Paste text below and press Predict</div></div>"
            "<div class='readout'>Fast • Lightweight</div></div>", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])

with col1:
    input_email = st.text_area("Enter email message", value=st.session_state.get("example_text",""), height=220)
    st.write("")  # spacing
    predict_btn = st.button("🔍 Predict", key="predict")

with col2:
    st.markdown("### Result")
    placeholder = st.empty()
    st.write("")
    st.markdown("### Info")
    st.markdown("- Model: RandomForest (trained in notebook)")
    st.markdown("- Preprocessing: lower, tokenize, remove stopwords, stem")

if predict_btn:
    if not input_email.strip():
        st.warning("Please enter a message to classify.")
    else:
        try:
            tfidf, model = load_artifacts()
        except Exception as e:
            st.error(f"Error loading model files: {e}")
            st.stop()

        with st.spinner("Preprocessing and predicting..."):
            x = transform_text(input_email)
            try:
                vector_input = tfidf.transform([x])
            except Exception as e:
                st.error(f"Vectorizer transform failed: {e}")
                st.stop()

            # get probability if available
            try:
                proba = model.predict_proba(vector_input)[0]
                spam_prob = float(proba[1]) if len(proba) > 1 else 0.0
            except Exception:
                # fallback to deterministic prediction
                spam_prob = 1.0 if int(model.predict(vector_input)[0]) == 1 else 0.0

            label = "Spam" if spam_prob >= 0.2 else "Ham"
            color = "🔴 Spam" if label == "Spam" else "🟢 Ham"
            percent = f"{spam_prob*100:.2f}%"

        # Display results in right column area
        placeholder.markdown(f"<div class='card'><h3 style='margin:0'>{color}</h3>"
                             f"<p class='small-muted' style='margin:0'>Confidence: <strong>{percent}</strong></p></div>",
                             unsafe_allow_html=True)

        # Show additional details
        st.markdown("### Prediction details")
        st.write(f"Label: **{label}**")
        st.write(f"Spam probability: **{percent}**")

        # quick suggestion / action
        if label == "Spam":
            st.error("This message is likely spam. Consider marking it as spam in your mail client.")
        else:
            st.success("This message looks like ham (not spam).")

st.markdown("</div>", unsafe_allow_html=True)

