from flask import Flask, redirect, url_for, render_template, request, session, flash
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database/history.sqlite3"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
history = db.Table("history", db.metadata, autoload=True, autoload_with=db.engine)
result = db.session.query(history).all()

PAINTING_NAME_COLUMN = 1
paintings = []

for i in range(len(result)):
    painting_name = result[i][PAINTING_NAME_COLUMN]
    if painting_name not in paintings:
        paintings.append(painting_name)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/query", methods=["POST", "GET"])
def query():
    if request.method == "POST":
        user_query = request.form["user_query"]
        # code 307 is used to redirect maintaining the same POST/GET method
        return redirect(url_for("query_result", your_query=user_query), code=307)
    else:
        return render_template("query.html", paintings=paintings)

@app.route("/query_result_<your_query>", methods=["POST", "GET"])
def query_result(your_query):
    if request.method == "POST":
        return render_template("query_result.html", content=your_query)
    else:
        return redirect(url_for("query"))


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
