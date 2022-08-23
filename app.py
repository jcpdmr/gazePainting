from flask import Flask, redirect, url_for, render_template, request, session, flash
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database/history.sqlite3"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['SECRET_KEY'] = "UnifiTesiIngETL"
db = SQLAlchemy(app)
history = db.Table("history", db.metadata, autoload=True, autoload_with=db.engine)
result = db.session.query(history).all()

PAINTING_NAME_COLUMN = 1
paintings = []
paintings_names = ""
for i in range(len(result)):
    painting_name = result[i][PAINTING_NAME_COLUMN]
    if painting_name not in paintings:
        paintings_names += painting_name + ", "
        paintings.append(painting_name)
paintings_names = paintings_names.strip()
paintings_names = paintings_names[:-1]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/query", methods=["POST", "GET"])
def query():
    if request.method == "POST":
        session["requested_painting"] = request.form["painting"]
        session["requested_gender"] = request.form["gender"]
        session["requested_gender_accuracy"] = request.form["gender_accuracy"]
        session["requested_age"] = request.form["age"]
        session["requested_age_accuracy"] = request.form["age_accuracy"]
        # code 307 is used to redirect maintaining the same POST/GET method
        return redirect(url_for("query_result"), code=307)
    else:
        return render_template("query.html", paintings=paintings)


@app.route("/query_result", methods=["POST", "GET"])
def query_result():
    if request.method == "POST":
        content = session.get("requested_painting")
        return render_template("query_result.html")  # TODO: content non viene mostrato da query_result.html
    else:
        return redirect(url_for("query"))


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
