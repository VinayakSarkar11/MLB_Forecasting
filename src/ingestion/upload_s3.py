"""
Upload locally partitioned Statcast CSVs from data/raw/ to S3.
S3 key structure: raw/year=YYYY/month=MM/statcast_YYYY_MM.csv
"""

import os
from pathlib import Path
import boto3
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

S3_BUCKET = os.environ["S3_BUCKET_NAME"]
S3_PREFIX = "raw"


def upload_all() -> None:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_REGION"],
    )

    csv_files = sorted(RAW_DIR.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found under {RAW_DIR}. Run extract.py first.")
        return

    for local_path in csv_files:
        # Preserve the year=/month= partition structure in the S3 key
        relative = local_path.relative_to(RAW_DIR)
        s3_key = f"{S3_PREFIX}/{relative}"

        print(f"Uploading {relative} → s3://{S3_BUCKET}/{s3_key}")
        s3.upload_file(str(local_path), S3_BUCKET, s3_key)

    print(f"\nDone: {len(csv_files)} file(s) uploaded to s3://{S3_BUCKET}/{S3_PREFIX}/")


if __name__ == "__main__":
    upload_all()
