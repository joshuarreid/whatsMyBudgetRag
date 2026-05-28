from __future__ import annotations

import unittest
from unittest.mock import Mock

from app.clients.spring_boot_client import SpringBootClient


class SpringBootClientAccountBlendingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = SpringBootClient()
        self.client.session = Mock()

    def _mock_json_response(self, payload):
        response = Mock()
        response.ok = True
        response.status_code = 200
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        return response

    def test_get_category_breakdown_passes_account_filter_through_without_joint_blending(self) -> None:
        self.client.session.get.return_value = self._mock_json_response(
            [
                {"category": "Dining Out", "totalAmount": "60.63", "transactionCount": 6},
                {"category": "Groceries", "totalAmount": "15.25", "transactionCount": 1},
            ]
        )

        payload = self.client.get_category_breakdown(period="December2025", account="josh")

        assert isinstance(payload, list)
        self.assertEqual(payload[0]["category"], "Dining Out")
        self.assertEqual(payload[0]["totalAmount"], "60.63")
        self.assertEqual(payload[0]["transactionCount"], 6)
        self.assertEqual(payload[1]["category"], "Groceries")
        self.assertEqual(payload[1]["totalAmount"], "15.25")
        self.assertEqual(payload[1]["transactionCount"], 1)
        self.assertEqual(self.client.session.get.call_count, 1)
        _, kwargs = self.client.session.get.call_args
        self.assertEqual(kwargs["params"]["account"], "josh")

    def test_get_period_overview_trusts_backend_summary_for_account_filters(self) -> None:
        self.client.session.get.return_value = self._mock_json_response(
            {
                "statementPeriod": "December2025",
                "paymentMethod": None,
                "account": "josh",
                "totalAmount": "3274.22",
                "transactionCount": 102,
            }
        )

        payload = self.client.get_period_overview(period="December2025", account="josh")

        self.assertEqual(str(payload["totalAmount"]), "3274.22")
        self.assertEqual(payload["transactionCount"], 102)
        self.assertEqual(payload["account"], "josh")
        self.assertEqual(self.client.session.get.call_count, 1)

    def test_get_period_overview_keeps_joint_account_unblended(self) -> None:
        self.client.session.get.return_value = self._mock_json_response(
            {
                "statementPeriod": "December2025",
                "paymentMethod": None,
                "account": "joint",
                "totalAmount": "3225.55",
                "transactionCount": 85,
            }
        )

        payload = self.client.get_period_overview(period="December2025", account="joint")

        self.assertEqual(str(payload["totalAmount"]), "3225.55")
        self.assertEqual(payload["transactionCount"], 85)
        self.assertEqual(self.client.session.get.call_count, 1)


if __name__ == "__main__":
    unittest.main()

