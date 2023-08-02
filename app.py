from flask import Flask, render_template, request, redirect
import utils

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")

    if request.method == 'POST':
        result = request.form.to_dict(flat=True)
        sentence_1 = result.get("sentence-1")
        sentence_2 = result.get("sentence-2")
        sim_msg = utils.similarity_value(sentence_1, sentence_2)
        result["sim_msg"] = sim_msg
        return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)

# testetstt