# evaluate_agreement.py
import argparse
import csv
import json
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path

from ner_common import (
    LABELS,
    load_texts,
    read_annotations,
    entity_set,
    bio_labels_for_doc,
    validate_rows,
)


def pairwise_entity_f1(rows_a, rows_b):
    """
    Exact-match entity-level precision/recall/F1.
    Match = same doc_id, start, end, and label.
    """
    A = entity_set(rows_a)
    B = entity_set(rows_b)
    tp = len(A & B)

    if len(A) == 0:
        precision = 1.0 if len(B) == 0 else 0.0
    else:
        precision = tp / len(A)

    if len(B) == 0:
        recall = 1.0 if len(A) == 0 else 0.0
    else:
        recall = tp / len(B)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 1.0 if len(A) == 0 and len(B) == 0 else 0.0

    return {
        "tp": tp,
        "n_a": len(A),
        "n_b": len(B),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def cohens_kappa(pairs):
    """
    Cohen's Kappa for a list of (label_a, label_b) token-level pairs.
    """
    n = len(pairs)
    if n == 0:
        return 0.0

    categories = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    matrix = [[0] * k for _ in range(k)]

    for a, b in pairs:
        matrix[idx[a]][idx[b]] += 1

    po = sum(matrix[i][i] for i in range(k)) / n
    row_sums = [sum(matrix[i]) for i in range(k)]
    col_sums = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row_sums[i] / n) * (col_sums[i] / n) for i in range(k))

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def build_bio_labels(rows, texts, doc_ids):
    """
    Returns dict: doc_id -> list of BIO labels for one annotator.
    If an annotator has no rows for a doc, all tokens are O.
    """
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)

    result = {}
    for doc_id in doc_ids:
        text = texts[doc_id]
        result[doc_id] = bio_labels_for_doc(by_doc.get(doc_id, []), doc_id, text)
    return result


def disagreement_details(rows_a, rows_b, ann_a, ann_b):
    """
    Pairwise disagreement details.
    Types:
      - category: same span, different label
      - boundary: overlapping span, same label, different boundaries
      - missing_extra: one annotator has an entity, the other has no overlapping same-label entity
    """
    by_doc_a = defaultdict(list)
    by_doc_b = defaultdict(list)
    for r in rows_a:
        by_doc_a[r["doc_id"]].append(r)
    for r in rows_b:
        by_doc_b[r["doc_id"]].append(r)

    A = entity_set(rows_a)
    B = entity_set(rows_b)
    details = []

    def add_side_only(side_rows, other_rows, side_name, other_name):
        for r in side_rows:
            key = (r["doc_id"], r["start"], r["end"], r["label"])
            if key in (A if side_name == ann_a else B):
                # Only process entities not exactly matched in the other annotation.
                other_set = B if side_name == ann_a else A
                if key in other_set:
                    continue

            doc_id = r["doc_id"]
            start, end, label = r["start"], r["end"], r["label"]

            same_span_other = [
                o for o in other_rows
                if o["doc_id"] == doc_id and o["start"] == start and o["end"] == end
            ]
            if same_span_other:
                for o in same_span_other:
                    if o["label"] != label:
                        details.append({
                            "pair": f"{ann_a} vs {ann_b}",
                            "doc_id": doc_id,
                            "type": "category",
                            "annotator_has": side_name,
                            "other_annotator": other_name,
                            "start": start,
                            "end": end,
                            "label": label,
                            "other_start": o["start"],
                            "other_end": o["end"],
                            "other_label": o["label"],
                            "note": f"{side_name}={label}; {other_name}={o['label']}",
                        })
                continue

            overlapping_same_label = [
                o for o in other_rows
                if o["doc_id"] == doc_id
                and o["label"] == label
                and o["start"] < end
                and o["end"] > start
            ]
            if overlapping_same_label:
                for o in overlapping_same_label:
                    details.append({
                        "pair": f"{ann_a} vs {ann_b}",
                        "doc_id": doc_id,
                        "type": "boundary",
                        "annotator_has": side_name,
                        "other_annotator": other_name,
                        "start": start,
                        "end": end,
                        "label": label,
                        "other_start": o["start"],
                        "other_end": o["end"],
                        "other_label": o["label"],
                        "note": f"{side_name} [{start}:{end}] vs {other_name} [{o['start']}:{o['end']}]",
                    })
            else:
                details.append({
                    "pair": f"{ann_a} vs {ann_b}",
                    "doc_id": doc_id,
                    "type": "missing_extra",
                    "annotator_has": side_name,
                    "other_annotator": other_name,
                    "start": start,
                    "end": end,
                    "label": label,
                    "other_start": "",
                    "other_end": "",
                    "other_label": "",
                    "note": f"{side_name} has {r['entity_text']!r}; {other_name} has no overlapping same-label entity",
                })

    add_side_only(rows_a, rows_b, ann_a, ann_b)
    add_side_only(rows_b, rows_a, ann_b, ann_a)
    return details


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER inter-annotator agreement.")
    parser.add_argument("--texts", required=True, help="Directory with one .txt file per doc_id")
    parser.add_argument("--annotations", nargs="+", required=True, help="Annotator CSV files")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    if len(args.annotations) < 2:
        raise ValueError("Need at least two annotation files.")

    texts = load_texts(args.texts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotators = {}
    all_rows = {}
    for path in args.annotations:
        name = Path(path).stem
        rows = read_annotations(path)
        errors, warnings = validate_rows(rows, texts)
        if errors:
            print(f"\nValidation errors in {path}:")
            for e in errors:
                print("ERROR:", e)
            raise SystemExit("Fix validation errors before evaluating.")
        annotators[name] = rows
        all_rows[name] = rows

    names = list(annotators.keys())
    doc_ids = sorted({r["doc_id"] for rows in annotators.values() for r in rows})

    # Precompute BIO labels per annotator/doc.
    bio = {
        name: build_bio_labels(annotators[name], texts, doc_ids)
        for name in names
    }

    f1_rows = []
    kappa_rows = []
    all_disagreements = []

    f1_values = []
    kappa_values = []

    for ann_a, ann_b in combinations(names, 2):
        f1_result = pairwise_entity_f1(annotators[ann_a], annotators[ann_b])
        f1_values.append(f1_result["f1"])
        f1_rows.append({
            "annotator_a": ann_a,
            "annotator_b": ann_b,
            **f1_result,
        })

        token_pairs = []
        for doc_id in doc_ids:
            la = bio[ann_a][doc_id]
            lb = bio[ann_b][doc_id]
            if len(la) != len(lb):
                raise ValueError(f"Token length mismatch for {doc_id}: {ann_a} vs {ann_b}")
            token_pairs.extend(zip(la, lb))

        kappa = cohens_kappa(token_pairs)
        kappa_values.append(kappa)
        kappa_rows.append({
            "annotator_a": ann_a,
            "annotator_b": ann_b,
            "n_token_pairs": len(token_pairs),
            "cohens_kappa": kappa,
        })

        all_disagreements.extend(
            disagreement_details(annotators[ann_a], annotators[ann_b], ann_a, ann_b)
        )

    # Save pairwise F1
    with open(out_dir / "pairwise_entity_f1.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["annotator_a", "annotator_b", "tp", "n_a", "n_b", "precision", "recall", "f1"],
        )
        writer.writeheader()
        writer.writerows(f1_rows)

    # Save pairwise kappa
    with open(out_dir / "pairwise_cohens_kappa.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["annotator_a", "annotator_b", "n_token_pairs", "cohens_kappa"],
        )
        writer.writeheader()
        writer.writerows(kappa_rows)

    # Save disagreements
    disagreement_fields = [
        "pair", "doc_id", "type", "annotator_has", "other_annotator",
        "start", "end", "label", "other_start", "other_end", "other_label", "note",
    ]
    with open(out_dir / "disagreements.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=disagreement_fields)
        writer.writeheader()
        writer.writerows(all_disagreements)

    disagreement_counts = Counter(d["type"] for d in all_disagreements)

    summary = {
        "annotators": names,
        "documents": doc_ids,
        "n_pairwise_comparisons": len(f1_rows),
        "average_exact_match_entity_f1": sum(f1_values) / len(f1_values),
        "average_cohens_kappa_token_bio": sum(kappa_values) / len(kappa_values),
        "pairwise_f1": f1_rows,
        "pairwise_kappa": kappa_rows,
        "disagreement_counts": dict(disagreement_counts),
        "n_disagreement_examples": len(all_disagreements),
    }

    with open(out_dir / "agreement_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Agreement summary ===")
    print(f"Annotators: {names}")
    print(f"Documents: {len(doc_ids)}")
    print(f"Pairwise comparisons: {len(f1_rows)}")
    print(f"Average exact-match entity-level F1: {summary['average_exact_match_entity_f1']:.4f}")
    print(f"Average token-level Cohen's Kappa: {summary['average_cohens_kappa_token_bio']:.4f}")
    print(f"Disagreement types: {dict(disagreement_counts)}")
    print(f"\nResults written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()