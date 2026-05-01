from __future__ import annotations

import os
from pathlib import Path

from minio import Minio


def get_minio_client() -> Minio:
    endpoint = os.environ["MINIO_ENDPOINT"]
    access_key = os.environ["MINIO_ACCESS_KEY"]
    secret_key = os.environ["MINIO_SECRET_KEY"]
    secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"

    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


def ensure_bucket(bucket: str) -> None:
    client = get_minio_client()

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def upload_file_to_minio(
    local_path: str | Path,
    bucket: str,
    object_name: str,
) -> None:
    local_path = Path(local_path)

    ensure_bucket(bucket)

    client = get_minio_client()
    client.fput_object(
        bucket_name=bucket,
        object_name=object_name,
        file_path=str(local_path),
    )


def download_file_from_minio(
    bucket: str,
    object_name: str,
    local_path: str | Path,
) -> Path:
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    client = get_minio_client()
    client.fget_object(
        bucket_name=bucket,
        object_name=object_name,
        file_path=str(local_path),
    )

    return local_path