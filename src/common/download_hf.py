from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a Hugging Face repository into a local directory.")
    parser.add_argument("repo_id")
    parser.add_argument("local_dir")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("HF_DOWNLOAD_MAX_WORKERS", "1")))
    parser.add_argument("--etag-timeout", type=int, default=int(os.environ.get("HF_HUB_ETAG_TIMEOUT", "120")))
    parser.add_argument("--download-timeout", type=int, default=int(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "120")))
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(args.etag_timeout))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(args.download_timeout))

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=args.max_workers,
        etag_timeout=args.etag_timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
