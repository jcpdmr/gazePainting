from flask import Flask, redirect, url_for, render_template, request, session, flash
from flask_sqlalchemy import SQLAlchemy
import json
from other_files.utils import getInfoPerPainting, getPaintingNames

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database/history.sqlite3"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['SECRET_KEY'] = "UnifiTesiIngETL"
db = SQLAlchemy(app)
history = db.Table("history", db.metadata, autoload=True, autoload_with=db.engine)

result = db.session.query(history).all()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboards")
def dashboards():
    painting_names = getPaintingNames(result)
    total_seconds_male, total_seconds_female, total_seconds, single_interactions_male, \
    single_interactions_female, single_interactions_total, age_interval_0_array_total,\
        age_interval_1_array_total, age_interval_2_array_total, age_interval_3_array_total,\
        age_interval_4_array_total, age_interval_5_array_total, age_interval_6_array_total, \
        age_interval_7_array_total = getInfoPerPainting(result)
    return render_template("dashboards.html", painting_names=json.dumps(painting_names),
                           total_seconds_male=json.dumps(total_seconds_male),
                           total_seconds_female=json.dumps(total_seconds_female),
                           total_seconds=json.dumps(total_seconds),
                           single_interactions_male=json.dumps(single_interactions_male),
                           single_interactions_female=json.dumps(single_interactions_female),
                           single_interactions_total=json.dumps(single_interactions_total),
                           age_interval_0_array_total=json.dumps(age_interval_0_array_total),
                           age_interval_1_array_total=json.dumps(age_interval_1_array_total),
                           age_interval_2_array_total=json.dumps(age_interval_2_array_total),
                           age_interval_3_array_total=json.dumps(age_interval_3_array_total),
                           age_interval_4_array_total=json.dumps(age_interval_4_array_total),
                           age_interval_5_array_total=json.dumps(age_interval_5_array_total),
                           age_interval_6_array_total=json.dumps(age_interval_6_array_total),
                           age_interval_7_array_total=json.dumps(age_interval_7_array_total))

@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
