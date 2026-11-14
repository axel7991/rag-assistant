
import hashlib

from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from fastapi.testclient import TestClient
from main import app


def test_upload_pdf(tmp_path):
    fake_chunks = [
        {
            "page_number": 2,
            "chunk_index": 0,
            "text": "The vehicle has a damaged front bumper."
        }
    ]

    fake_embedding = [0.1, 0.2, 0.3]

    test_uuid = UUID("12345678-1234-5678-1234-567812345678")
    pdf_content = b"%PDF-1.4 fake test content"

    with (
        patch("main.extract_pdf_chunks", return_value=fake_chunks),
        patch("main.embedding_model") as mock_model,
        patch("main.collection") as mock_collection,
        patch("main.uuid4", return_value=test_uuid),
        patch("main.shutil.copyfileobj"),
        patch("builtins.open", create=True)
    ):
        mock_model.encode.return_value.tolist.return_value = [
            fake_embedding
        ]

        client = TestClient(app)

        response = client.post(
            "/upload",
            files={
                "file": (
                    "sample.pdf",
                    pdf_content,
                    "application/pdf"
                )
            }
        )

        assert response.status_code == 200
        assert response.json()["total_chunks"] == 1

        expected_file_hash = hashlib.sha256(
            pdf_content
        ).hexdigest()

        expected_chunk_id = hashlib.sha256(
            f"{expected_file_hash}:2:0".encode()
        ).hexdigest()

        mock_collection.upsert.assert_called_once_with(
            ids=[expected_chunk_id],
            embeddings=[fake_embedding],
            documents=[
                "The vehicle has a damaged front bumper."
            ],
            metadatas=[
                {
                    "filename": "sample.pdf",
                    "page_number": 2,
                    "chunk_index": 0
                }
            ]
        )


def test_upload_rejects_non_pdf():
    client = TestClient(app)

    response = client.post(
        "/upload",
        files={
            "file": (
                "notes.txt",
                b"This is a text document.",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Only PDF files are allowed"
    )


def test_upload_rejects_oversized_pdf():
    client = TestClient(app)

    oversized_content = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/upload",
        files={
            "file": (
                "large.pdf",
                oversized_content,
                "application/pdf"
            )
        }
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "File exceeds the 10 MB limit"
    )


def test_upload_uses_safe_filename(tmp_path):
    test_uuid = UUID("12345678-1234-5678-1234-567812345678")

    with (
        patch("main.Path.mkdir"),
        patch("main.uuid4", return_value=test_uuid),
        patch("main.extract_pdf_chunks", return_value=[]),
        patch("main.shutil.copyfileobj"),
        patch("builtins.open", create=True) as mock_open
    ):
        client = TestClient(app)

        response = client.post(
            "/upload",
            files={
                "file": (
                    "../../malicious.pdf",
                    b"%PDF-1.4 test",
                    "application/pdf"
                )
            }
        )

        assert response.status_code == 200

        saved_path = mock_open.call_args.args[0]

        # Works on both Windows and Linux
        assert Path(saved_path) == (
            Path("uploads")
            / "12345678123456781234567812345678.pdf"
        )

        # Verify directory traversal was prevented
        assert ".." not in Path(saved_path).parts


def test_upload_multiple_chunks():
    fake_chunks = [
        {
            "page_number": 1,
            "chunk_index": 0,
            "text": "Front bumper damaged."
        },
        {
            "page_number": 2,
            "chunk_index": 1,
            "text": "Rear door scratched."
        }
    ]

    fake_embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]

    test_uuid = UUID("12345678-1234-5678-1234-567812345678")
    pdf_content = b"%PDF-1.4 test"

    with (
        patch("main.extract_pdf_chunks", return_value=fake_chunks),
        patch("main.embedding_model") as mock_model,
        patch("main.collection") as mock_collection,
        patch("main.uuid4", return_value=test_uuid),
        patch("main.shutil.copyfileobj"),
        patch("builtins.open", create=True)
    ):
        mock_model.encode.return_value.tolist.return_value = (
            fake_embeddings
        )

        client = TestClient(app)

        response = client.post(
            "/upload",
            files={
                "file": (
                    "vehicle.pdf",
                    pdf_content,
                    "application/pdf"
                )
            }
        )

        assert response.status_code == 200
        assert response.json()["total_chunks"] == 2

        mock_model.encode.assert_called_once_with(
            [
                "Front bumper damaged.",
                "Rear door scratched."
            ],
            batch_size=32
        )

        mock_collection.upsert.assert_called_once()

        kwargs = mock_collection.upsert.call_args.kwargs

        expected_file_hash = hashlib.sha256(
            pdf_content
        ).hexdigest()

        expected_ids = [
            hashlib.sha256(
                f"{expected_file_hash}:1:0".encode()
            ).hexdigest(),
            hashlib.sha256(
                f"{expected_file_hash}:2:1".encode()
            ).hexdigest()
        ]

        assert kwargs["ids"] == expected_ids

        assert kwargs["documents"] == [
            "Front bumper damaged.",
            "Rear door scratched."
        ]

        assert kwargs["embeddings"] == fake_embeddings

        assert kwargs["metadatas"] == [
            {
                "filename": "vehicle.pdf",
                "page_number": 1,
                "chunk_index": 0
            },
            {
                "filename": "vehicle.pdf",
                "page_number": 2,
                "chunk_index": 1
            }
        ]
