"""Command line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import data, pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demoscout", description="Train, audit and package the release screener.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="full pipeline")
    run.add_argument("--data", type=Path, default=data.DEFAULT_PATH)
    run.add_argument("--out", type=Path, default=Path("outputs"))
    run.add_argument("--assets", type=Path, default=Path("assets"))
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--boot", type=int, default=300)
    sub.add_parser("fetch", help="download the csv")
    app = sub.add_parser("app", help="launch the gradio screener")
    app.add_argument("--bundle", type=Path, default=Path("outputs/scout.joblib"))
    args = parser.parse_args(argv)
    if args.cmd == "fetch":
        print(data.fetch())
    elif args.cmd == "run":
        res = pipeline.run(args.out, args.assets, args.data, args.seed, args.boot)
        print(f"shipped range model: {res['range_selection']['shipped']}")
        print(f"shipped hit model: {res['hit_selection']['shipped']}")
        print(f"hit AUC on unseen artists: {res['hit_test'][res['hit_selection']['shipped']]['auc']}")
    else:
        from .app import launch

        launch(args.bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
