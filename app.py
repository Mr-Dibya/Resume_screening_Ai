from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Store history
history = []

@app.route("/", methods=["GET", "POST"])
def home():
    global history
    results = []
    top_candidate = ""
    top_score = 0

    if request.method == "POST":
        job_description = request.form["job"]
        resumes = request.form["resumes"].split("\n")

        documents = resumes + [job_description]

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(vectors)
        scores = similarity[-1][:-1]

        for i, score in enumerate(scores):
            results.append({
                "resume": resumes[i],
                "score": round(score * 100, 2),
                "status": "Selected" if score > 0.3 else "Not Selected"
            })

        best_index = scores.argmax()
        top_candidate = resumes[best_index]
        top_score = round(scores[best_index] * 100, 2)

        # Save to history
        history.append({
            "job": job_description,
            "top_candidate": top_candidate,
            "score": top_score
        })

    return render_template(
        "index.html",
        results=results,
        top_candidate=top_candidate,
        top_score=top_score,
        history=history
    )

#Clear history route
@app.route("/clear")
def clear_history():
    global history
    history = []
    return render_template("index.html", results=[], history=[])

if __name__ == "__main__":
    app.run(debug=True)
