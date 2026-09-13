-- Applied by the platform, in filename order, as the app's own role.
-- Must stay idempotent: it runs on every start.

CREATE TABLE IF NOT EXISTS prompts (
  id         bigserial   PRIMARY KEY,
  prompt     text        NOT NULL,
  answer     text        NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
