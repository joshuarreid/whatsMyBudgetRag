-- =========================================================
-- budget_rag: conversation history + tool caches + retention
-- MySQL 8.0+
-- =========================================================

SET NAMES utf8mb4;
SET time_zone = '+00:00';

CREATE DATABASE IF NOT EXISTS budget_rag
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE budget_rag;

-- -----------------------------
-- 1) Conversations
-- -----------------------------
CREATE TABLE IF NOT EXISTS conversations (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_uuid CHAR(36) NOT NULL,
  user_id VARCHAR(128) NOT NULL,
  title VARCHAR(255) NULL,
  status ENUM('active','archived','deleted') NOT NULL DEFAULT 'active',
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  last_message_at DATETIME(6) NULL,
  metadata JSON NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uk_conversations_uuid (conversation_uuid),
  KEY idx_conversations_user_updated (user_id, updated_at DESC),
  KEY idx_conversations_status_updated (status, updated_at DESC),
  CONSTRAINT chk_conversations_metadata_json CHECK (metadata IS NULL OR JSON_VALID(metadata))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------
-- 2) Messages
-- -----------------------------
CREATE TABLE IF NOT EXISTS messages (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  message_uuid CHAR(36) NOT NULL,
  conversation_id BIGINT UNSIGNED NOT NULL,
  sequence_no INT UNSIGNED NOT NULL,

  role ENUM('system','user','assistant','tool') NOT NULL,
  content LONGTEXT NOT NULL,
  content_type ENUM('text','markdown','json') NOT NULL DEFAULT 'text',

  transaction_id VARCHAR(128) NULL,
  request_id VARCHAR(128) NULL,
  model_name VARCHAR(128) NULL,
  prompt_tokens INT UNSIGNED NULL,
  completion_tokens INT UNSIGNED NULL,
  total_tokens INT UNSIGNED NULL,
  latency_ms INT UNSIGNED NULL,

  period VARCHAR(32) NULL,
  period_source VARCHAR(64) NULL,
  tool_plan JSON NULL,
  context_json JSON NULL,
  answer_json JSON NULL,

  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  edited_at DATETIME(6) NULL,
  is_deleted TINYINT(1) NOT NULL DEFAULT 0,

  PRIMARY KEY (id),
  UNIQUE KEY uk_messages_uuid (message_uuid),
  UNIQUE KEY uk_messages_conversation_sequence (conversation_id, sequence_no),
  KEY idx_messages_conversation_created (conversation_id, created_at),
  KEY idx_messages_conversation_role_created (conversation_id, role, created_at),
  KEY idx_messages_transaction (transaction_id),

  CONSTRAINT fk_messages_conversation
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
    ON DELETE CASCADE,

  CONSTRAINT chk_messages_tool_plan_json CHECK (tool_plan IS NULL OR JSON_VALID(tool_plan)),
  CONSTRAINT chk_messages_context_json CHECK (context_json IS NULL OR JSON_VALID(context_json)),
  CONSTRAINT chk_messages_answer_json CHECK (answer_json IS NULL OR JSON_VALID(answer_json))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------
-- 3) Message tool calls
-- -----------------------------
CREATE TABLE IF NOT EXISTS message_tool_calls (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  message_id BIGINT UNSIGNED NOT NULL,
  tool_name VARCHAR(128) NOT NULL,
  call_order SMALLINT UNSIGNED NOT NULL DEFAULT 1,
  arguments_json JSON NULL,
  result_json JSON NULL,
  status ENUM('ok','error','timeout') NOT NULL DEFAULT 'ok',
  error_text TEXT NULL,
  duration_ms INT UNSIGNED NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),

  PRIMARY KEY (id),
  KEY idx_tool_calls_message_order (message_id, call_order),
  KEY idx_tool_calls_tool_created (tool_name, created_at),

  CONSTRAINT fk_tool_calls_message
    FOREIGN KEY (message_id) REFERENCES messages(id)
    ON DELETE CASCADE,

  CONSTRAINT chk_tool_calls_arguments_json CHECK (arguments_json IS NULL OR JSON_VALID(arguments_json)),
  CONSTRAINT chk_tool_calls_result_json CHECK (result_json IS NULL OR JSON_VALID(result_json))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------
-- 4) Message citations
-- -----------------------------
CREATE TABLE IF NOT EXISTS message_citations (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  message_id BIGINT UNSIGNED NOT NULL,
  citation_order SMALLINT UNSIGNED NOT NULL DEFAULT 1,
  source_type ENUM('api','document','sql','other') NOT NULL DEFAULT 'api',
  source_ref VARCHAR(512) NOT NULL,
  source_title VARCHAR(255) NULL,
  snippet TEXT NULL,
  score DECIMAL(6,5) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),

  PRIMARY KEY (id),
  KEY idx_citations_message_order (message_id, citation_order),

  CONSTRAINT fk_citations_message
    FOREIGN KEY (message_id) REFERENCES messages(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------
-- 5) Conversation-scoped tool cache
-- -----------------------------
CREATE TABLE IF NOT EXISTS conversation_tool_cache (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT UNSIGNED NOT NULL,
  tool_name VARCHAR(128) NOT NULL,
  cache_key CHAR(64) NOT NULL,
  period VARCHAR(32) NULL,
  params_json JSON NOT NULL,
  response_json JSON NOT NULL,
  source_message_id BIGINT UNSIGNED NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  expires_at DATETIME(6) NULL,
  invalidated_at DATETIME(6) NULL,
  hit_count INT UNSIGNED NOT NULL DEFAULT 0,
  last_hit_at DATETIME(6) NULL,

  PRIMARY KEY (id),
  UNIQUE KEY uk_conv_tool_cache (conversation_id, tool_name, cache_key),
  KEY idx_cache_lookup (conversation_id, tool_name, period, expires_at),
  KEY idx_cache_expiry (expires_at),
  KEY idx_cache_source_message (source_message_id),

  CONSTRAINT fk_cache_conversation
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
    ON DELETE CASCADE,

  CONSTRAINT fk_cache_source_message
    FOREIGN KEY (source_message_id) REFERENCES messages(id)
    ON DELETE SET NULL,

  CONSTRAINT chk_cache_params_json CHECK (JSON_VALID(params_json)),
  CONSTRAINT chk_cache_response_json CHECK (JSON_VALID(response_json))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------
-- 6) Shared tool-result cache
-- -----------------------------
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

-- -----------------------------
-- 7) Optional view for timeline replay
-- -----------------------------
CREATE OR REPLACE VIEW v_conversation_messages AS
SELECT
  c.conversation_uuid,
  c.user_id,
  m.message_uuid,
  m.sequence_no,
  m.role,
  m.content,
  m.content_type,
  m.period,
  m.period_source,
  m.transaction_id,
  m.request_id,
  m.model_name,
  m.created_at
FROM conversations c
JOIN messages m ON m.conversation_id = c.id
WHERE m.is_deleted = 0;

-- -----------------------------
-- 8) 30-day retention event
-- -----------------------------
-- Requires EVENT privilege. The server administrator must enable the event scheduler
-- outside this script if it is disabled (for example in my.cnf or with a privileged
-- admin session). This script avoids privileged GLOBAL variable changes.

DROP EVENT IF EXISTS ev_cleanup_conversation_retention_30d;

DELIMITER $$

CREATE EVENT ev_cleanup_conversation_retention_30d
ON SCHEDULE EVERY 1 DAY
STARTS CURRENT_TIMESTAMP + INTERVAL 5 MINUTE
DO
BEGIN
  -- Cache cleanup
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

  -- History cleanup (cascade removes dependent rows)
  DELETE FROM conversations
  WHERE COALESCE(last_message_at, created_at) < (UTC_TIMESTAMP(6) - INTERVAL 30 DAY);
END$$

DELIMITER ;

-- Verify setup
-- SHOW VARIABLES LIKE 'event_scheduler';
-- SHOW EVENTS LIKE 'ev_cleanup_conversation_retention_30d';

