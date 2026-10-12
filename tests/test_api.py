import unittest

from fastapi.testclient import TestClient
from app.main import app


class TestTicketAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Entering the context runs startup and loads the model.
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_health(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_prediction_structure(self):
        response = self.client.post(
            "/predict",
            json={"text": "I need a copy of my invoice."},
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()

        self.assertIn(
            result["suggested_category"],
            app.state.pipeline.classes_.tolist(),
        )
        self.assertEqual(len(result["top_categories"]), 3)
        self.assertTrue(result["requires_human_review"])

    def test_scores_are_sorted_and_bounded(self):
        response = self.client.post(
            "/predict",
            json={"text": "Help me update my account."},
        )

        scores = [
            item["score"]
            for item in response.json()["top_categories"]
        ]

        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0 <= score <= 1 for score in scores))

    def test_empty_text(self):
        response = self.client.post(
            "/predict", json={"text": ""}
        )
        self.assertEqual(response.status_code, 422)

    def test_whitespace_text(self):
        response = self.client.post(
            "/predict", json={"text": "   "}
        )
        self.assertEqual(response.status_code, 422)

    def test_missing_text(self):
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 422)

    def test_oversized_text(self):
        response = self.client.post(
            "/predict", json={"text": "x" * 5001}
        )
        self.assertEqual(response.status_code, 422)

    def test_non_string_text(self):
        response = self.client.post(
            "/predict", json={"text": 123}
        )
        self.assertEqual(response.status_code, 422)

    def test_homepage(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "text/html",
            response.headers["content-type"],
        )
        self.assertIn(
            "Support Ticket Classifier",
            response.text,
        )

    def test_homepage_displays_limitations(self):
        response = self.client.get("/")

        self.assertIn("synthetic messages", response.text)
        self.assertIn(
            "Suggestions require human review",
            response.text,
        )


if __name__ == "__main__":
    unittest.main()
