"""Automated Defect Detection package.

Ensures the MySQL schema exists on first import by calling
`init_db()` in a safe, idempotent way. This lets UI scripts that
import `automated_defect_detection.*` work without a separate
initializer step.
"""

try:
    from .database_manager import init_db  # local import to avoid global side-effects
    # Best-effort DB initialization (creates database and tables if missing)
    try:
        init_db()
    except Exception:
        # Keep imports working even if DB is temporarily unavailable.
        # Errors are logged inside init_db; swallow here to avoid breaking UI startup.
        pass
except Exception:
    # If even importing the DB layer fails (e.g., connector missing), we still
    # allow package import so UI can display a friendly error later.
    pass
