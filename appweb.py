from flask import Flask, render_template
import sqlite3
import json
from helpers_files.utils import getInfoPerPainting, getPaintingNames

app = Flask(__name__)
app.config['SECRET_KEY'] = "UnifiTesiIngETL"

def get_db():
    """Return a SQLite connection and cursor."""
    conn = sqlite3.connect('database/history.db')
    conn.row_factory = sqlite3.Row
    return conn, conn.cursor()

@app.route("/")
def home():
    """Render the home page."""
    return render_template("home.html")

@app.route("/dashboards")
def dashboards():
    """Render the dashboards page."""
    # Get a connection and cursor
    conn, cur = get_db()

    # Execute the query to fetch data from the database
    cur.execute("SELECT * FROM history")
    result = cur.fetchall()

    # Close the connection
    conn.close()

    # Utilize the obtained data to compute the required results
    painting_names = getPaintingNames(result)
    total_seconds_male, total_seconds_female, total_seconds, single_interactions_male, \
    single_interactions_female, single_interactions_total, age_interval_0_array_total,\
        age_interval_1_array_total, age_interval_2_array_total, age_interval_3_array_total,\
        age_interval_4_array_total, age_interval_5_array_total, age_interval_6_array_total, \
        age_interval_7_array_total = getInfoPerPainting(result)

    # Pass the results to the HTML page
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
    """Render the about page."""
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
