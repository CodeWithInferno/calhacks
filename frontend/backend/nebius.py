"""
Optional Nebius helper for the inference service.

Used when your real policy needs to (a) load a trained checkpoint from the bucket
or (b) cache generated rollouts back to it. Credentials are read from the env /
AWS profile by boto3's default chain — never hardcoded, never sent to the browser.

The stub does not call this; it's here so the integration seam in inference.py
has a ready-made, credential-safe S3 client.
"""

import os


def make_s3_client():
    """Return a boto3 S3 client pointed at Nebius Object Storage."""
    import boto3  # imported lazily so the stub runs without boto3 installed

    endpoint = os.environ["NEBIUS_S3_ENDPOINT"]
    region = os.environ["NEBIUS_REGION"]
    force_path = os.environ.get("NEBIUS_S3_FORCE_PATH_STYLE", "false").lower() == "true"

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        config=boto3.session.Config(
            s3={"addressing_style": "path" if force_path else "virtual"}
        ),
        # Credentials resolved by boto3's default chain:
        #   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or AWS_PROFILE=nebius.
    )


def download_bytes(key: str) -> bytes:
    bucket = os.environ["NEBIUS_BUCKET"]
    obj = make_s3_client().get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()
