from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "final_model.joblib"
)

model = joblib.load(MODEL_PATH)


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>GUPIO - Student Academic Outcome Prediction</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f7fb;
            margin: 0;
            padding: 30px;
        }

        .container {
            max-width: 900px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }

        h1 {
            text-align: center;
            color: #1f3c88;
        }

        .subtitle {
            text-align: center;
            color: #555;
            margin-bottom: 25px;
        }

        .info {
            background: #eef4ff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        button {
            width: 100%;
            padding: 14px;
            background: #1f3c88;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 17px;
            cursor: pointer;
        }

        button:hover {
            background: #162d68;
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            background: #e8f5e9;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #176b2c;
        }

        .error {
            margin-top: 20px;
            padding: 15px;
            background: #ffecec;
            color: #b00020;
            border-radius: 8px;
        }

        .field {
            margin-bottom: 15px;
        }

        label {
            display: block;
            font-weight: bold;
            margin-bottom: 5px;
        }

        input {
            width: 100%;
            padding: 10px;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 6px;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>GUPIO</h1>

    <div class="subtitle">
        Student Academic Outcome Prediction
    </div>

    <div class="info">
        Enter student academic information below and click
        <b>Predict Academic Outcome</b>.
    </div>

    <form method="POST">

        {% for feature, value in fields %}
        <div class="field">
            <label>{{ feature }}</label>
            <input
                type="number"
                step="any"
                name="{{ feature }}"
                value="{{ value }}"
                required
            >
        </div>
        {% endfor %}

        <button type="submit">
            Predict Academic Outcome
        </button>

    </form>

    {% if prediction %}
    <div class="result">
        Predicted Outcome: {{ prediction }}
    </div>
    {% endif %}

    {% if error %}
    <div class="error">
        Error: {{ error }}
    </div>
    {% endif %}

</div>

</body>
</html>
"""


# These are the 24 non-leakage input features
FEATURES = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]


# Example values
DEFAULT_VALUES = [
    1,
    17,
    1,
    9254,
    1,
    1,
    130,
    1,
    19,
    37,
    9,
    9,
    125,
    1,
    0,
    0,
    1,
    0,
    0,
    20,
    0,
    10.8,
    1.4,
    1.74
]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    fields = list(zip(FEATURES, DEFAULT_VALUES))

    if request.method == "POST":

        try:

            data = {}

            for feature in FEATURES:
                data[feature] = float(request.form[feature])

            input_data = pd.DataFrame([data])

            prediction = model.predict(input_data)[0]

        except Exception as e:

            error = str(e)

    return render_template_string(
        HTML,
        fields=fields,
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )