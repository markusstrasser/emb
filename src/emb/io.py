"""JSONL reading/writing for emb entry format."""
import json
import sys
from pathlib import Path
from typing import List
from emb.schema import Entry, validate_entry


def read_jsonl(source) -> List[Entry]:
    """Read entries from JSONL file or stdin.

    Args:
        source: Path to .jsonl file, or '-' for stdin, or file-like object
    """
    entries = []
    close_after = False

    if source == '-':
        fh = sys.stdin
    elif hasattr(source, 'read'):
        fh = source
    else:
        fh = open(source, 'r')
        close_after = True

    try:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid JSON on line {line_num}: {e}", file=sys.stderr)
                continue
            if not validate_entry(d):
                print(f"Warning: skipping line {line_num} (missing id or text)", file=sys.stderr)
                continue
            entries.append(Entry.from_dict(d))
    finally:
        if close_after:
            fh.close()

    return entries


def write_jsonl(entries: List[Entry], dest) -> int:
    """Write entries as JSONL. Returns count written."""
    close_after = False

    if dest == '-':
        fh = sys.stdout
    elif hasattr(dest, 'write'):
        fh = dest
    else:
        fh = open(dest, 'w')
        close_after = True

    count = 0
    try:
        for entry in entries:
            json.dump(entry.to_dict(), fh, ensure_ascii=False)
            fh.write('\n')
            count += 1
    finally:
        if close_after:
            fh.close()

    return count
