
import sqlite3
import pytest

from database import init_db


def test_database_initialization(tmp_path, monkeypatch):
    test_db = tmp_path / "test_rag.db"

    monkeypatch.setattr(
        "database.DATABASE_PATH",
        test_db
    )

    init_db()

    connection = sqlite3.connect(test_db)

    try:
        cursor = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name = 'uploads'
            """
        )

        result = cursor.fetchone()

        assert result is not None
        assert result[0] == "uploads"

    finally:
        connection.close()


def test_duplicate_idempotency_key(tmp_path, monkeypatch):
    test_db = tmp_path / "test_rag.db"

    monkeypatch.setattr(
        "database.DATABASE_PATH",
        test_db
    )

    init_db()

    connection = sqlite3.connect(test_db)

    try:
        connection.execute(
            """
            INSERT INTO uploads
                (idempotency_key, filename, file_hash)
            VALUES (?, ?, ?)
            """,
            ("upload-123", "vehicle.pdf", "hash123")
        )

        connection.commit()

        # A duplicate key must be rejected by SQLite.
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO uploads
                    (idempotency_key, filename, file_hash)
                VALUES (?, ?, ?)
                """,
                ("upload-123", "vehicle.pdf", "hash123")
            )

    finally:
        connection.close()

def test_get_upload_by_key(tmp_path, monkeypatch):
    test_db = tmp_path / "test_rag.db"

    monkeypatch.setattr(
        "database.DATABASE_PATH",
        test_db
    )

    init_db()

    with sqlite3.connect(test_db) as connection:
        connection.execute(
            """
            INSERT INTO uploads
                (idempotency_key, filename, file_hash, status)
            VALUES (?, ?, ?, ?)
            """,
            ("upload-123", "vehicle.pdf", "hash123", "completed")
        )

    from database import get_upload_by_key

    upload = get_upload_by_key("upload-123")

    assert upload is not None
    assert upload["idempotency_key"] == "upload-123"
    assert upload["filename"] == "vehicle.pdf"
    assert upload["file_hash"] == "hash123"
    assert upload["status"] == "completed"

    assert get_upload_by_key("unknown-key") is None
