from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


class MainAppCorsTests(unittest.TestCase):
    def _settings(self, *, cors_enabled: bool, cors_allowed_origins: tuple[str, ...]) -> Settings:
        return Settings(
            spring_boot_base_url="http://springboot-api",
            request_timeout_seconds=10,
            log_level="INFO",
            log_format="text",
            cors_enabled=cors_enabled,
            cors_allowed_origins=cors_allowed_origins,
            openai_api_key=None,
            openai_chat_model="gpt-4o-mini",
            mysql_host=None,
            mysql_port=25060,
            mysql_database=None,
            mysql_user=None,
            mysql_password=None,
            mysql_ssl_disabled=False,
            mysql_ssl_ca=None,
            mysql_sslmode=None,
            mysql_connect_timeout_seconds=10,
            conversation_default_user="default-user",
            conversation_history_context_limit=10,
            insight_high_share_threshold=45,
            insight_outlier_amount_threshold=500,
        )

    def test_cors_preflight_allows_configured_origin_when_enabled(self) -> None:
        app = create_app(
            self._settings(
                cors_enabled=True,
                cors_allowed_origins=("http://localhost:3000",),
            )
        )

        with TestClient(app) as client:
            response = client.options(
                "/rag/ask",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:3000")

    def test_cors_preflight_has_no_allow_origin_header_when_disabled(self) -> None:
        app = create_app(self._settings(cors_enabled=False, cors_allowed_origins=("http://localhost:3000",)))

        with TestClient(app) as client:
            response = client.options(
                "/rag/ask",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                },
            )

        self.assertEqual(response.status_code, 405)
        self.assertIsNone(response.headers.get("access-control-allow-origin"))


if __name__ == "__main__":
    unittest.main()

