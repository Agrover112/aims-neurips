"""Create the shared Cloudflare R2 bucket and register it as a Modal Secret.

The bucket is the hand-off point between Modal workspaces, which cannot share
Volumes. This script creates it once and writes the same ``aims-r2`` Secret into
every listed Modal profile, so ``modal_bucket_transfer.py`` runs from either.

Credentials come from the environment, never from arguments, so they stay out of
shell history and process listings. The R2 API token needs *Admin Read & Write*
because creating a bucket is an account-level action.

    export R2_ACCOUNT_ID=...  R2_ACCESS_KEY_ID=...  R2_SECRET_ACCESS_KEY=...
    uv run --with boto3 python scripts/setup_r2_bucket.py \
        --bucket aims-neurips-transfer --profiles agrover112 aims-neurips
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

SECRET_NAME = "aims-r2"

_CREATE_SECRET = """
import os, modal
modal.Secret.objects.create(
    os.environ["SECRET_NAME"],
    {k: os.environ[k] for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "R2_ENDPOINT")},
    allow_existing=True,
)
"""


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Set {name} in the environment first.")
    return value


def create_bucket(bucket: str, endpoint: str, key_id: str, secret: str) -> None:
    import boto3
    from botocore.exceptions import ClientError

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name="auto",
    )
    try:
        client.create_bucket(Bucket=bucket)
        print(f"created bucket {bucket}")
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
            raise
        print(f"bucket {bucket} already exists")


def create_secret(profile: str, key_id: str, secret: str, endpoint: str) -> None:
    env = {
        **os.environ,
        "MODAL_PROFILE": profile,
        "SECRET_NAME": SECRET_NAME,
        "AWS_ACCESS_KEY_ID": key_id,
        "AWS_SECRET_ACCESS_KEY": secret,
        "R2_ENDPOINT": endpoint,
    }
    subprocess.run([sys.executable, "-I", "-c", _CREATE_SECRET], env=env, check=True)
    print(f"secret {SECRET_NAME} ready in profile {profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bucket", default="aims-neurips-transfer")
    parser.add_argument("--profiles", nargs="+", default=["agrover112", "aims-neurips"])
    args = parser.parse_args()

    endpoint = f"https://{_require('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com"
    key_id = _require("R2_ACCESS_KEY_ID")
    secret = _require("R2_SECRET_ACCESS_KEY")

    create_bucket(args.bucket, endpoint, key_id, secret)
    for profile in args.profiles:
        create_secret(profile, key_id, secret, endpoint)
    print(f"\nexport R2_BUCKET={args.bucket}  R2_ENDPOINT={endpoint}")


if __name__ == "__main__":
    main()
