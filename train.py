import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Example dataset
data = [
    # Fake news
    ("Drinking bleach cures COVID-19 and other diseases.", 0),
    ("Government secretly implanted microchips in all new smartphones.", 0),
    ("Aliens are residing in the U.S. and helping their government for world domination.", 0),
    ("Eating chocolate every day will make you live 20 years longer.", 0),
    ("New study proves the Earth is flat and NASA is lying.", 0),

    # Real news
    ("The United Nations held a meeting to discuss climate change policies.", 1),
    ("Apple announced the release of the new iPhone 16 with upgraded cameras.", 1),
    ("India successfully launched its Mars orbiter into orbit.", 1),
    ("Stock markets fell slightly today due to global economic concerns.", 1),
    ("Local school introduces new online learning programs for students.", 1),
]

# Split data
texts, labels = zip(*data)

# Text preprocessing function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # remove non-letters
    text = text.lower().strip()
    return text

cleaned_texts = [clean_text(t) for t in texts]

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_texts)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X, labels)

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
