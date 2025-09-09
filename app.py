import streamlit as st
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ----------------- Preprocessing -----------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

# ----------------- Load Saved Model & Vectorizer -----------------
model = pickle.load(open("fake_news_model.pkl", "rb"))
vector = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.write("Enter a news headline or content below and check if it's **Real** or **Fake**.")

user_input = st.text_area("✍️ Type or paste news text here:")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before checking!")
    else:
        # Preprocess input
        processed = stemming(user_input)
        vectorized = vector.transform([processed])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.error("❌ This news looks **Fake**")
        else:
            st.success("✅ This news looks **Real**")
