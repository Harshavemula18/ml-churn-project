
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "customer_churn.csv"
HTML_PATH = ROOT / "index.html"

FEATURES = [
    "monthly_charge",
    "months_as_customer",
    "support_tickets",
    "weekly_usage_hours",
]


def train_model():
    data = pd.read_csv(CSV_PATH)

    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].fillna(data[column].median())

    data["churned"] = pd.to_numeric(data["churned"], errors="coerce")
    data = data.dropna(subset=["churned"])

    if not set(data["churned"].unique()).issubset({0, 1}):
        raise ValueError("churned must contain only 0 and 1")

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(data[FEATURES], data["churned"].astype(int))
    return model


model = train_model()


class handler(BaseHTTPRequestHandler):

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            try:
                body = HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header(
                    "Content-Type", "text/html; charset=utf-8"
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                self.send_json(
                    500, {"error": "Could not load the website homepage"}
                )
        else:
            self.send_json(
                200, {"message": "Customer Churn Prediction API is running"}
            )

    def do_POST(self):
        if self.path.split("?")[0] != "/api/predict":
            self.send_json(404, {"error": "Endpoint not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 10000:
                self.send_json(400, {"error": "Invalid request"})
                return

            payload = json.loads(
                self.rfile.read(length).decode("utf-8")
            )

            values = {}

            for feature in FEATURES:
                value = float(payload[feature])

                if (
                    not pd.notna(value)
                    or value < 0
                    or value == float("inf")
                    or value == float("-inf")
                ):
                    self.send_json(
                        400,
                        {"error": f"{feature} must be a non-negative number"},
                    )
                    return

                if (
                    feature in ("months_as_customer", "support_tickets")
                    and not value.is_integer()
                ):
                    self.send_json(
                        400,
                        {"error": f"{feature} must be a whole number"},
                    )
                    return

                values[feature] = value

            customer = pd.DataFrame([values], columns=FEATURES)
            prediction = int(model.predict(customer)[0])

            self.send_json(
                200,
                {
                    "prediction": prediction,
                    "result": (
                        "Likely to churn"
                        if prediction == 1
                        else "Likely to stay"
                    ),
                    "explanation": (
                        "The model predicts this customer may cancel."
                        if prediction == 1
                        else "The model predicts this customer may stay."
                    ),
                },
            )

        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            self.send_json(
                400,
                {"error": "Enter valid values for all four fields"},
            )
        except Exception:
            self.send_json(
                500,
                {"error": "Prediction failed. Check the server logs."},
            )
