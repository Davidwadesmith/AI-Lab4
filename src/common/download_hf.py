from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description="Download a Hugging Face repository into a local directory.")
    parser.add_argument("repo_id")
    parser.add_argument("local_dir")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    args = parser.parse_args(argv)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
