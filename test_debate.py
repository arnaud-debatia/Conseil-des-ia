import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import debate


class DebateTests(unittest.TestCase):
    def test_paid_and_unknown_prices_are_ineligible(self):
        self.assertFalse(debate.eligible({"pricing": {"prompt": "0"}}))
        self.assertFalse(debate.eligible({"pricing": {"prompt": "0", "completion": "0", "request": "1"}}))
        self.assertTrue(debate.eligible({"pricing": {"prompt": "0", "completion": "0"}}))

    def test_paid_model_is_never_called(self):
        with patch.object(debate, "request") as request:
            self.assertEqual(debate.ask("paid/model", "test")["status"], "error")
            request.assert_not_called()

    def test_free_price_cap_and_model_identity(self):
        model = debate.CONFIG["models"][0]
        response = {"id": "test-id", "model": "other/model", "choices": [{"message": {"content": "test"}}]}
        with patch.object(debate, "request", return_value=response) as request:
            self.assertEqual(debate.ask(model, "test")["status"], "error")
            self.assertEqual(request.call_args.args[1]["provider"]["max_price"], debate.PRICES)

    def test_replies_receive_other_models_positions(self):
        models = debate.CONFIG["models"]
        with patch.object(debate, "ask", side_effect=lambda model, prompt: {
            "model": model, "status": "ok", "answer": model, "prompt": prompt,
        }):
            session = debate.run_session(1, models, [])
        for row in session["rounds"]["repliques"]:
            self.assertTrue(all(model in row["prompt"] for model in models))

    def test_final_session_receives_previous_replies(self):
        models = debate.CONFIG["models"]
        history = [{"day": 1, "question": "Sujet initial", "rounds": {
            "repliques": [{"model": models[0], "status": "ok", "answer": "Réplique antérieure"}],
        }}]
        with patch.object(debate, "ask", side_effect=lambda model, prompt: {
            "model": model, "status": "ok", "answer": prompt,
        }):
            session = debate.run_session(7, models, history)
        self.assertIn("Réplique antérieure", session["rounds"]["positions"][0]["answer"])

    def test_check_does_not_call_api(self):
        with patch("sys.argv", ["debate.py", "check"]), patch.object(debate, "request") as request:
            debate.main()
            request.assert_not_called()

    def test_existing_day_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "jour-1.json").write_text("{}", encoding="utf-8")
            date = debate.datetime.now(debate.ZoneInfo("Europe/Paris")).date().isoformat()
            with patch.object(debate, "DATA", data), patch.dict(os.environ, {"CONSEIL_START_DATE": date}), patch(
                "sys.argv", ["debate.py", "debate"]
            ), patch.object(debate, "request") as request:
                debate.main()
                request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
