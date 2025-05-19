import os
from flask import Flask, request, render_template
import whisper
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    transcript = None

    if request.method == "POST":
        file = request.files["audio_file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            transcript = whisper.load_model("base").transcribe(filepath)["text"]

            prompt = f"Summarize the following meeting transcript:\n\n{transcript}"
            response = model.generate_content(prompt)
            summary = response.text

    return render_template("index.html", summary=summary, transcript=transcript)

if __name__ == "__main__":
    app.run(debug=True)
