
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json  

DATABASE_PATH = Path("rag_metadata.db")


def get_connection():
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30
    )
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'processing'
                    CHECK(status IN (
                        'processing',
                        'completed',
                        'failed'
                    )),
                response_json TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()


def get_upload_by_key(idempotency_key: str):
    """
    Find an upload by its idempotency key.

    Returns a dictionary if found, otherwise None.
    """
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                id,
                idempotency_key,
                filename,
                file_hash,
                status,
                response_json,
                error_message,
                created_at,
                updated_at
            FROM uploads
            WHERE idempotency_key = ?
            """,
            (idempotency_key,)
        ).fetchone()

    return dict(row) if row is not None else None



def reserve_upload(idempotency_key: str, filename: str, file_hash: str) -> bool:
    """
    Atomically reserve an idempotency key.

    Returns True if the key was reserved successfully.
    Returns False if the key already exists.
    """
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO uploads
                (idempotency_key, filename, file_hash, status)
            VALUES (?, ?, ?, 'processing')
            """,
            (idempotency_key, filename, file_hash)
        )
        connection.commit()

        return cursor.rowcount == 1


def recover_stale_upload(
    idempotency_key: str,
    file_hash: str,
    stale_after_minutes: int = 10
) -> bool:
    """
    Atomically reclaim an upload that has been stuck
    in 'processing' longer than the allowed timeout.

    Returns True only when this request successfully
    claims the stale upload.
    """

    if stale_after_minutes <= 0:
        raise ValueError("stale_after_minutes must be positive")

    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(minutes=stale_after_minutes)
    ).strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE uploads
            SET status = 'processing',
                error_message = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE idempotency_key = ?
              AND file_hash = ?
              AND status = 'processing'
              AND updated_at < ?
            """,
            (idempotency_key, file_hash, cutoff)
        )

        connection.commit()
        return cursor.rowcount == 1




def complete_upload(idempotency_key: str, response: dict) -> bool:
    response_json = json.dumps(response)

    with get_connection() as connection:
        cursor = connection.execute("""
            UPDATE uploads
            SET status = 'completed',
                response_json = ?,
                error_message = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE idempotency_key = ?
              AND status = 'processing'
        """, (response_json, idempotency_key))

        connection.commit()
        return cursor.rowcount == 1

def fail_upload(idempotency_key: str, error_message: str) -> bool:
    with get_connection() as connection:
        cursor = connection.execute("""
            UPDATE uploads
            SET status = 'failed',
                error_message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE idempotency_key = ?
              AND status = 'processing'
        """, (error_message, idempotency_key))

        connection.commit()
        return cursor.rowcount == 1



if __name__ == "__main__":
    init_db()
    print("SQLite database initialized successfully!")
