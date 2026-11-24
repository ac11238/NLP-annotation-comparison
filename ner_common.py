import csv
import re
from pathlib import Path
from collections import defaultdict

LABELS = {"PER", "ORG", "LOC"}

# Canonical field names and accepted aliases
FIELD_ALIASES = {
    "doc_id": {"doc_id", "document_id", "document id", "docid", "doc"},
    "entity_text": {"entity_text", "entity text", "text", "entity", "mention"},
    "label": {"label", "category", "entity_type", "type"},
    "start": {"start", "start_offset", "start offset", "begin"},
    "end": {"end", "end_offset", "end offset", "stop"},
}


def _norm_field(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower())


ALIAS_TO_CANON = {}
for canon, aliases in FIELD_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_CANON[_norm_field(alias)] = canon


def load_texts(text_dir: str) -> dict:
    """
    Load one .txt file per document.
    The filename stem is the doc_id, e.g. texts/news_001.txt -> doc_id "news_001".
    """
    texts = {}
    for path in Path(text_dir).glob("*.txt"):
        texts[path.stem] = path.read_text(encoding="utf-8")
    if not texts:
        raise ValueError(f"No .txt files found in {text_dir}")
    return texts


def read_annotations(csv_path: str) -> list:
    """
    Read an annotation CSV and normalize column names.
    Returns a list of dicts with keys: doc_id, entity_text, label, start, end, _file, _line
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path}: empty or missing header")

        field_map = {}
        for fname in reader.fieldnames:
            canon = ALIAS_TO_CANON.get(_norm_field(fname))
            if canon:
                field_map[fname] = canon

        missing = set(FIELD_ALIASES) - set(field_map.values())
        if missing:
            raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

        for line_no, raw in enumerate(reader, start=2):
            rec = {canon: raw[fname] for fname, canon in field_map.items()}
            rec["doc_id"] = rec["doc_id"].strip()
            rec["entity_text"] = rec["entity_text"]  # keep exact; do not strip
            rec["label"] = rec["label"].strip().upper()
            rec["start"] = int(rec["start"])
            rec["end"] = int(rec["end"])
            rec["_file"] = str(csv_path)
            rec["_line"] = line_no
            rows.append(rec)
    return rows


def entity_key(row: dict) -> tuple:
    """Exact-match key for entity-level evaluation."""
    return (row["doc_id"], row["start"], row["end"], row["label"])


def entity_set(rows: list) -> set:
    return {entity_key(r) for r in rows}


def tokenize_with_offsets(text: str) -> list:
    """
    Simple deterministic tokenizer:
      - word tokens: letters/digits/underscore
      - single non-space punctuation tokens
    Returns list of (token_text, start, end).
    """
    return [
        (m.group(0), m.start(), m.end())
        for m in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    ]


def validate_rows(rows: list, texts: dict) -> tuple:
    """
    Validate one annotator's rows.
    Returns (errors, warnings).
    """
    errors = []
    warnings = []
    by_doc = defaultdict(list)
    seen = set()

    for r in rows:
        loc = f"{r['_file']}:{r['_line']}"
        doc_id = r["doc_id"]
        start = r["start"]
        end = r["end"]
        label = r["label"]
        entity_text = r["entity_text"]

        if label not in LABELS:
            errors.append(f"{loc}: invalid label {label!r}; expected one of {sorted(LABELS)}")
            continue

        if doc_id not in texts:
            errors.append(f"{loc}: doc_id {doc_id!r} not found in texts directory")
            continue

        text = texts[doc_id]

        if not (0 <= start < end <= len(text)):
            errors.append(
                f"{loc}: invalid offsets start={start}, end={end}, "
                f"text length={len(text)}"
            )
            continue

        actual = text[start:end]
        if actual != entity_text:
            errors.append(
                f"{loc}: entity_text mismatch. CSV={entity_text!r}, "
                f"text[{start}:{end}]={actual!r}"
            )

        key = (doc_id, start, end, label)
        if key in seen:
            warnings.append(f"{loc}: duplicate annotation {key}")
        seen.add(key)

        by_doc[doc_id].append(r)

    # Check overlaps within each annotator/document.
    # Proposal says: do not annotate nested or overlapping entities.
    for doc_id, doc_rows in by_doc.items():
        ordered = sorted(doc_rows, key=lambda x: (x["start"], x["end"]))
        for prev, cur in zip(ordered, ordered[1:]):
            if cur["start"] < prev["end"]:
                errors.append(
                    f"{doc_id}: overlapping entities within one annotation file: "
                    f"{prev['entity_text']!r} [{prev['start']}:{prev['end']}] "
                    f"and {cur['entity_text']!r} [{cur['start']}:{cur['end']}]"
                )

    return errors, warnings


def bio_labels_for_doc(rows_for_doc: list, doc_id: str, text: str) -> list:
    """
    Convert entity annotations for one document into token-level BIO labels.
    Assumes no overlapping entities; validate_annotations.py should catch overlaps.
    """
    tokens = tokenize_with_offsets(text)
    labels = ["O"] * len(tokens)

    for r in sorted(rows_for_doc, key=lambda x: (x["start"], x["end"])):
        ent_start, ent_end = r["start"], r["end"]
        ent_label = r["label"]

        covered = [
            i for i, (_, tok_start, tok_end) in enumerate(tokens)
            if tok_start < ent_end and tok_end > ent_start
        ]

        if not covered:
            continue

        first = covered[0]
        if labels[first] != "O":
            raise ValueError(
                f"Token-label conflict in {doc_id} at token {first}: "
                f"cannot assign B-{ent_label}"
            )
        labels[first] = f"B-{ent_label}"

        for i in covered[1:]:
            if labels[i] not in ("O", f"I-{ent_label}"):
                raise ValueError(
                    f"Token-label conflict in {doc_id} at token {i}: "
                    f"cannot assign I-{ent_label}"
                )
            labels[i] = f"I-{ent_label}"

    return labels