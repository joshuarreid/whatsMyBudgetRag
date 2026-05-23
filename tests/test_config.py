from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.core.config import get_settings


class SettingsTests(unittest.TestCase):
    def tearDown(self) -> None:
        get_settings.cache_clear()

    def test_mysql_and_conversation_settings_load_from_environment(self) -> None:
        env = {
            "MYSQL_HOST": "db.example.com",
            "MYSQL_PORT": "25060",
            "MYSQL_DATABASE": "budget_rag",
            "MYSQL_USER": "doadmin",
            "MYSQL_PASSWORD": "secret",
            "MYSQL_SSL_DISABLED": "false",
            "MYSQL_SSL_CA": "/tmp/ca.pem",
            "MYSQL_CONNECT_TIMEOUT_SECONDS": "12",
            "CONVERSATION_DEFAULT_USER": "default-user",
            "CONVERSATION_HISTORY_CONTEXT_LIMIT": "8",
        }
        with patch.dict(os.environ, env, clear=False):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.mysql_host, "db.example.com")
        self.assertEqual(settings.mysql_port, 25060)
        self.assertEqual(settings.mysql_database, "budget_rag")
        self.assertEqual(settings.mysql_user, "doadmin")
        self.assertEqual(settings.mysql_password, "secret")
        self.assertFalse(settings.mysql_ssl_disabled)
        self.assertEqual(settings.mysql_ssl_ca, "/tmp/ca.pem")
        self.assertEqual(settings.mysql_connect_timeout_seconds, 12)
        self.assertEqual(settings.conversation_default_user, "default-user")
        self.assertEqual(settings.conversation_history_context_limit, 8)


if __name__ == "__main__":
    unittest.main()

