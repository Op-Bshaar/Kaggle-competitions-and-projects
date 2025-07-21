import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
nltk.data.path.append(os.path.abspath("nltk_data"))


intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
]


patterns = []
tags = []
for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = tags
clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X, y)


def chatbot(input_text):
    input_vector = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_vector)[0]
    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])


def handle_input():
    user_input = st.session_state.user_input
    if user_input:
        response = chatbot(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        st.session_state.user_input = ""


def main():
    st.title("💬 Chatbot")
    st.write("Type a message and press Enter")

  
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    
    st.text_input("You:", key="user_input", on_change=handle_input)

    
    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 Bot:** {msg}")

if __name__ == "__main__":
    main()
