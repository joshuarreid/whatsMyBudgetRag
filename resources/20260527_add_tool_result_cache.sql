-- Adds a shared tool_result_cache table for reusable historical analytics results
-- and refreshes the retention event so expired shared cache rows are cleaned up.

USE budget_rag;

CREATE TABLE IF NOT EXISTS tool_result_cache (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  tool_name VARCHAR(128) NOT NULL,
  cache_key CHAR(64) NOT NULL,
  scope_type VARCHAR(64) NULL,
  statement_period VARCHAR(32) NULL,
  start_period VARCHAR(32) NULL,
  end_period VARCHAR(32) NULL,
  start_date DATE NULL,
  end_date DATE NULL,
  account VARCHAR(128) NULL,
  payment_method VARCHAR(128) NULL,
  params_json JSON NOT NULL,
  response_json JSON NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  expires_at DATETIME(6) NULL,
  invalidated_at DATETIME(6) NULL,
  hit_count INT UNSIGNED NOT NULL DEFAULT 0,
  last_hit_at DATETIME(6) NULL,

  PRIMARY KEY (id),
  UNIQUE KEY uk_tool_result_cache (tool_name, cache_key),
  KEY idx_tool_result_cache_lookup (tool_name, expires_at, invalidated_at),
  KEY idx_tool_result_cache_scope_period (scope_type, statement_period),
  KEY idx_tool_result_cache_scope_dates (scope_type, start_date, end_date),
  KEY idx_tool_result_cache_account (account),
  KEY idx_tool_result_cache_payment_method (payment_method),
  KEY idx_tool_result_cache_expiry (expires_at),

  CONSTRAINT chk_tool_result_cache_params_json CHECK (JSON_VALID(params_json)),
  CONSTRAINT chk_tool_result_cache_response_json CHECK (JSON_VALID(response_json))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

DROP EVENT IF EXISTS ev_cleanup_conversation_retention_30d;

DELIMITER $$

CREATE EVENT ev_cleanup_conversation_retention_30d
ON SCHEDULE EVERY 1 DAY
STARTS CURRENT_TIMESTAMP + INTERVAL 5 MINUTE
DO
BEGIN
  DELETE FROM conversation_tool_cache
  WHERE
    invalidated_at IS NOT NULL
    OR (expires_at IS NOT NULL AND expires_at < UTC_TIMESTAMP(6))
    OR created_at < (UTC_TIMESTAMP(6) - INTERVAL 30 DAY);

  DELETE FROM tool_result_cache
  WHERE
    invalidated_at IS NOT NULL
    OR (expires_at IS NOT NULL AND expires_at < UTC_TIMESTAMP(6))
    OR created_at < (UTC_TIMESTAMP(6) - INTERVAL 30 DAY);

  DELETE FROM conversations
  WHERE COALESCE(last_message_at, created_at) < (UTC_TIMESTAMP(6) - INTERVAL 30 DAY);
END$$

DELIMITER ;

