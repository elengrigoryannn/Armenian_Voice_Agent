import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

RAW_DATA_DIR = "raw_data"
ALLOWED_TOPICS = ["loans", "deposits", "branches"]


def slugify(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--institution", required=True, help='e.g. "Ameriabank"')
    parser.add_argument("--topic", required=True, choices=ALLOWED_TOPICS)
    parser.add_argument("--url", required=True,
                         help="The official source page (used for citations), "
                              "or \"manual entry\" if there truly isn't one.")
    parser.add_argument("--title", default=None,
                         help="Optional page/section title. Defaults to "
                              "'<Institution> <Topic> (manual entry)'.")
    parser.add_argument("--file", default=None,
                         help="Path to a text file with the content. If "
                              "omitted, reads from stdin — paste the text "
                              "then send EOF (Ctrl+D, or Ctrl+Z+Enter on Windows).")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        print("Paste the content below, then press Ctrl+D (Ctrl+Z then Enter on Windows):\n")
        text = sys.stdin.read().strip()

    if not text:
        print("No text provided — nothing written.")
        return

    title = args.title or f"{args.institution} {args.topic} (manual entry)"

    record = {
        "institution": args.institution,
        "url": args.url,
        "title": title,
        "topic": args.topic,
        "text": text,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }

    inst_dir = os.path.join(RAW_DATA_DIR, args.institution)
    os.makedirs(inst_dir, exist_ok=True)
    filename = f"{args.topic}__manual_{slugify(args.url + str(datetime.now()))}.json"
    path = os.path.join(inst_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {path} ({len(text)} chars)")
    print("Next: python rag.py ingest")


if __name__ == "__main__":
    main()
