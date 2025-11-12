from __future__ import annotations
import argparse
from .pipeline import init_db, ingest, run_classify, run_decode, stats

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aws-mail-ingest")
    sp = ap.add_subparsers(dest="cmd", required=True)

    sp.add_parser("init")

    p_ing = sp.add_parser("ingest")
    p_ing.add_argument("--window", type=int, default=2000, help="UID window size per FETCH")
    p_ing.add_argument("--throttle-ms", type=int, default=0, help="sleep between windows")
    p_ing.add_argument("--start-uid", type=int, default=None, help="override checkpoint starting UID")
    p_ing.add_argument("--max-messages", type=int, default=None, help="stop after N messages (backfill batching)")

    p_cls = sp.add_parser("classify"); p_cls.add_argument("--limit", type=int, default=200)
    p_dec = sp.add_parser("decode");   p_dec.add_argument("--limit", type=int, default=200)
    sp.add_parser("stats")

    args = ap.parse_args(argv)

    if args.cmd == "init":
        init_db()
    elif args.cmd == "ingest":
        ingest(window=args.window, throttle_ms=args.throttle_ms, start_override=args.start_uid, max_messages=args.max_messages)
    elif args.cmd == "classify":
        run_classify(limit=args.limit)
    elif args.cmd == "decode":
        run_decode(limit=args.limit)
    elif args.cmd == "stats":
        stats()
    return 0