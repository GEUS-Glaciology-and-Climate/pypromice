from __future__ import annotations
import argparse
from .pipeline import init_db, ingest, run_classify, run_decode, stats

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aws-mail-ingest")
    sp = ap.add_subparsers(dest="cmd", required=True)

    sp.add_parser("init")
    sp.add_parser("ingest")
    p_cls = sp.add_parser("classify"); p_cls.add_argument("--limit", type=int, default=200)
    p_dec = sp.add_parser("decode");   p_dec.add_argument("--limit", type=int, default=200)
    sp.add_parser("stats")
    sp.add_parser("requeue-failed")  # reserved; implement if needed

    args = ap.parse_args(argv)

    if args.cmd == "init":
        init_db()
    elif args.cmd == "ingest":
        ingest()
    elif args.cmd == "classify":
        run_classify(limit=args.limit)
    elif args.cmd == "decode":
        run_decode(limit=args.limit)
    elif args.cmd == "stats":
        stats()
    elif args.cmd == "requeue-failed":
        print("Not implemented in split version. (Easy to add.)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())