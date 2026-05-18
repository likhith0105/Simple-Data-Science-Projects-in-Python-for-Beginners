# ==========================================
# SIMPLE NLP CHATBOT USING PYTHON
# ==========================================

import random
import nltk
from nltk.chat.util import Chat, reflections

# Download tokenizer data
nltk.download('punkt')

# ==========================================
# CHATBOT RESPONSES
# ==========================================

pairs = [

    [
        r"hi|hello|hey",
        ["Hello!", "Hi there!", "Hey!"]
    ],

    [
        r"what is your name ?",
        ["I am a Simple NLP Chatbot."]
    ],

    [
        r"how are you ?",
        ["I am fine. How about you?"]
    ],

    [
        r"i am fine",
        ["Great to hear that!"]
    ],

    [
        r"what can you do ?",
        ["I can chat with you and answer simple questions."]
    ],

    [
        r"who created you ?",
        ["I was created using Python and NLP."]
    ],

    [
        r"(.*) your favorite color ?",
        ["I like blue."]
    ],

    [
        r"(.*) your favorite food ?",
        ["I like digital data!"]
    ],

    [
        r"bye",
        ["Goodbye!", "See you later!", "Bye!"]
    ],

    [
        r"(.*)",
        ["Sorry, I did not understand that."]
    ]

]

# ==========================================
# CREATE CHATBOT
# ==========================================

chatbot = Chat(pairs, reflections)

# ==========================================
# START CHAT
# ==========================================

print("===================================")
print(" SIMPLE NLP CHATBOT ")
print(" Type 'bye' to exit ")
print("===================================")

chatbot.converse()