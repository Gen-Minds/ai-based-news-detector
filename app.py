from flask import Flask, render_template, request

app = Flask(__name__)

# Simple fake news keywords (for demo purposes)
fake_keywords = ["aliens", "cancer cure", "time travel", "ban internet", "underground base"]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form["news"].lower()
        if news_text.strip() != "":
            # simple keyword check
            if any(word in news_text for word in fake_keywords):
                prediction = "❌ Fake News"
            else:
                prediction = "✅ Real News"

    return render_template("index.html", prediction=prediction, news_text=news_text)

if __name__ == "__main__":
    app.run(debug=True)
