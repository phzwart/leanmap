#!/usr/bin/env python
"""Download PDB validation + crystallographic metadata for every structure.

Pulls from the RCSB Search + Data (GraphQL) APIs:

  - space group (Hermann–Mauguin + International Tables number)
  - unit cell (a, b, c, α, β, γ, volume, Z)
  - resolution
  - wwPDB validation metrics (MolProbity geometry, diffraction fit, EM/NMR)

Examples
--------
# smoke test
python examples/download_pdb_validation.py --limit 50 -o /tmp/pdb_val.csv

# full archive (~250k entries; resumable)
python examples/download_pdb_validation.py -o examples/data/pdb_validation.csv

# X-ray only
python examples/download_pdb_validation.py --method X-ray -o examples/data/pdb_xray.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
HOLDINGS_URL = "https://data.rcsb.org/rest/v1/holdings/current/entry_ids"

# Batch size for GraphQL `entries(entry_ids: ...)`. Keep modest — validation
# payloads are large and the API soft-caps at 1000 IDs.
DEFAULT_BATCH = 100

GRAPHQL_QUERY = """
query ($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    symmetry {
      Int_Tables_number
      space_group_name_H_M
      cell_setting
    }
    cell {
      length_a
      length_b
      length_c
      angle_alpha
      angle_beta
      angle_gamma
      volume
      Z_PDB
    }
    rcsb_entry_info {
      resolution_combined
      experimental_method
      structure_determination_methodology
    }
    refine {
      ls_d_res_high
      ls_R_factor_R_work
      ls_R_factor_R_free
      ls_percent_reflns_R_free
      correlation_coeff_Fo_to_Fc
    }
    pdbx_vrpt_summary {
      report_creation_date
      RNA_suiteness
    }
    pdbx_vrpt_summary_geometry {
      clashscore
      angles_RMSZ
      bonds_RMSZ
      percent_ramachandran_outliers
      percent_rotamer_outliers
    }
    pdbx_vrpt_summary_diffraction {
      DCC_R
      DCC_Rfree
      EDS_R
      EDS_res_high
      data_completeness
      Fo_Fc_correlation
      percent_RSRZ_outliers
      I_over_sigma
      twin_fraction
    }
    pdbx_vrpt_summary_em {
      Q_score
      atom_inclusion_all_atoms
      atom_inclusion_backbone
      calculated_fsc_resolution_by_cutoff_pt_143
    }
    pdbx_vrpt_summary_nmr {
      chemical_shift_completeness
      nmrclust_number_of_models
      nmrclust_number_of_outliers
    }
  }
}
""".strip()

FIELDNAMES = [
    "pdb_id",
    "experimental_method",
    "methodology",
    "resolution",
    "refine_resolution_high",
    "space_group_hm",
    "space_group_number",
    "cell_setting",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "volume",
    "Z_PDB",
    "r_work",
    "r_free",
    "percent_free_reflections",
    "cc_fo_fc",
    "clashscore",
    "angles_rmsz",
    "bonds_rmsz",
    "percent_ramachandran_outliers",
    "percent_rotamer_outliers",
    "dcc_r",
    "dcc_rfree",
    "eds_r",
    "eds_res_high",
    "data_completeness",
    "fo_fc_correlation",
    "percent_rsrz_outliers",
    "i_over_sigma",
    "twin_fraction",
    "em_q_score",
    "em_atom_inclusion_all",
    "em_atom_inclusion_backbone",
    "em_fsc_res_0p143",
    "nmr_cs_completeness",
    "nmrclust_n_models",
    "nmrclust_n_outliers",
    "rna_suiteness",
    "validation_report_date",
]


def _http_json(
    url: str,
    *,
    payload: Optional[dict] = None,
    timeout: float = 120.0,
    retries: int = 5,
) -> Any:
    """GET or POST JSON with exponential backoff on 429 / 5xx / network errors."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"

    last_err: Optional[BaseException] = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_err = exc
            body = exc.read().decode("utf-8", errors="replace")[:300]
            retryable = exc.code in {408, 425, 429, 500, 502, 503, 504}
            if not retryable or attempt == retries - 1:
                raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
            sleep = min(60.0, (2**attempt) + 0.25 * attempt)
            print(f"  retry {attempt + 1}/{retries} after HTTP {exc.code} ({sleep:.1f}s)", file=sys.stderr)
            time.sleep(sleep)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            if attempt == retries - 1:
                raise
            sleep = min(60.0, (2**attempt) + 0.25 * attempt)
            print(f"  retry {attempt + 1}/{retries} after {type(exc).__name__} ({sleep:.1f}s)", file=sys.stderr)
            time.sleep(sleep)
    raise RuntimeError(f"request failed: {last_err}")


def fetch_entry_ids(
    *,
    method: Optional[str] = None,
    page_size: int = 1000,
    limit: Optional[int] = None,
    cache_path: Optional[Path] = None,
) -> List[str]:
    """Return current experimental PDB IDs via the Search API (paginated).

    If ``cache_path`` is set, IDs are appended there after every page so a kill
    mid-listing does not throw away progress.
    """
    nodes: List[dict] = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.structure_determination_methodology",
                "operator": "exact_match",
                "value": "experimental",
            },
        }
    ]
    if method:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.experimental_method",
                    "operator": "exact_match",
                    "value": method,
                },
            }
        )

    query: dict = (
        nodes[0]
        if len(nodes) == 1
        else {"type": "group", "logical_operator": "and", "nodes": nodes}
    )

    ids: List[str] = []
    start = 0
    if cache_path is not None and cache_path.exists():
        ids = [x.strip().upper() for x in cache_path.read_text().splitlines() if x.strip()]
        start = len(ids)
        print(f"Resuming ID list from cache ({start:,} already listed)", file=sys.stderr)

    total: Optional[int] = None
    cache_fh = cache_path.open("a") if cache_path is not None else None
    try:
        while True:
            rows = page_size
            if limit is not None:
                remaining = limit - len(ids)
                if remaining <= 0:
                    break
                rows = min(rows, remaining)
            payload = {
                "query": query,
                "return_type": "entry",
                "request_options": {
                    "paginate": {"start": start, "rows": rows},
                    "results_content_type": ["experimental"],
                    "sort": [{"sort_by": "rcsb_id", "direction": "asc"}],
                },
            }
            data = _http_json(SEARCH_URL, payload=payload, timeout=180.0)
            if total is None:
                total = int(data.get("total_count", 0))
                target = f"{limit:,}" if limit is not None else f"{total:,}"
                print(f"Search API: {total:,} entries (fetching {target})", file=sys.stderr)
            batch = [hit["identifier"].upper() for hit in data.get("result_set", [])]
            if not batch:
                break
            ids.extend(batch)
            if cache_fh is not None:
                cache_fh.write("\n".join(batch) + "\n")
                cache_fh.flush()
            start += len(batch)
            print(f"  listed {len(ids):,}/{total:,}", file=sys.stderr)
            if limit is not None and len(ids) >= limit:
                return ids[:limit]
            if total is not None and start >= total:
                break
    finally:
        if cache_fh is not None:
            cache_fh.close()
    return ids


def fetch_entry_ids_holdings() -> List[str]:
    """Fallback: full current holdings list (can be slow / large)."""
    print("Fetching holdings list…", file=sys.stderr)
    data = _http_json(HOLDINGS_URL, timeout=300.0)
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected holdings payload: {type(data)}")
    print(f"Holdings: {len(data):,} entries", file=sys.stderr)
    return [str(x) for x in data]


def _as_dict(value: Any) -> dict:
    """Normalize GraphQL objects that may arrive as a dict, list, or null."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return value[0] if value and isinstance(value[0], dict) else {}
    return {}


def _one(value: Any) -> Any:
    """Unwrap single-element lists (e.g. resolution_combined)."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def flatten_entry(entry: Optional[dict]) -> Dict[str, Any]:
    if not entry:
        return {}
    info = _as_dict(entry.get("rcsb_entry_info"))
    sym = _as_dict(entry.get("symmetry"))
    cell = _as_dict(entry.get("cell"))
    refine = _as_dict(entry.get("refine"))
    geom = _as_dict(entry.get("pdbx_vrpt_summary_geometry"))
    diffr = _as_dict(entry.get("pdbx_vrpt_summary_diffraction"))
    em = _as_dict(entry.get("pdbx_vrpt_summary_em"))
    nmr = _as_dict(entry.get("pdbx_vrpt_summary_nmr"))
    summary = _as_dict(entry.get("pdbx_vrpt_summary"))

    pdb_id = entry.get("rcsb_id")
    return {
        "pdb_id": pdb_id.upper() if isinstance(pdb_id, str) else pdb_id,
        "experimental_method": info.get("experimental_method"),
        "methodology": info.get("structure_determination_methodology"),
        "resolution": _one(info.get("resolution_combined")),
        "refine_resolution_high": refine.get("ls_d_res_high"),
        "space_group_hm": sym.get("space_group_name_H_M"),
        "space_group_number": sym.get("Int_Tables_number"),
        "cell_setting": sym.get("cell_setting"),
        "a": cell.get("length_a"),
        "b": cell.get("length_b"),
        "c": cell.get("length_c"),
        "alpha": cell.get("angle_alpha"),
        "beta": cell.get("angle_beta"),
        "gamma": cell.get("angle_gamma"),
        "volume": cell.get("volume"),
        "Z_PDB": cell.get("Z_PDB"),
        "r_work": refine.get("ls_R_factor_R_work"),
        "r_free": refine.get("ls_R_factor_R_free"),
        "percent_free_reflections": refine.get("ls_percent_reflns_R_free"),
        "cc_fo_fc": refine.get("correlation_coeff_Fo_to_Fc"),
        "clashscore": geom.get("clashscore"),
        "angles_rmsz": geom.get("angles_RMSZ"),
        "bonds_rmsz": geom.get("bonds_RMSZ"),
        "percent_ramachandran_outliers": geom.get("percent_ramachandran_outliers"),
        "percent_rotamer_outliers": geom.get("percent_rotamer_outliers"),
        "dcc_r": diffr.get("DCC_R"),
        "dcc_rfree": diffr.get("DCC_Rfree"),
        "eds_r": diffr.get("EDS_R"),
        "eds_res_high": diffr.get("EDS_res_high"),
        "data_completeness": diffr.get("data_completeness"),
        "fo_fc_correlation": diffr.get("Fo_Fc_correlation"),
        "percent_rsrz_outliers": diffr.get("percent_RSRZ_outliers"),
        "i_over_sigma": diffr.get("I_over_sigma"),
        "twin_fraction": diffr.get("twin_fraction"),
        "em_q_score": em.get("Q_score"),
        "em_atom_inclusion_all": em.get("atom_inclusion_all_atoms"),
        "em_atom_inclusion_backbone": em.get("atom_inclusion_backbone"),
        "em_fsc_res_0p143": em.get("calculated_fsc_resolution_by_cutoff_pt_143"),
        "nmr_cs_completeness": nmr.get("chemical_shift_completeness"),
        "nmrclust_n_models": nmr.get("nmrclust_number_of_models"),
        "nmrclust_n_outliers": nmr.get("nmrclust_number_of_outliers"),
        "rna_suiteness": summary.get("RNA_suiteness"),
        "validation_report_date": summary.get("report_creation_date"),
    }


def fetch_batch(ids: Sequence[str]) -> List[dict]:
    data = _http_json(
        GRAPHQL_URL,
        payload={"query": GRAPHQL_QUERY, "variables": {"ids": list(ids)}},
        timeout=180.0,
    )
    if data.get("errors"):
        # Partial data is still useful; log and continue with whatever came back.
        msgs = "; ".join(e.get("message", str(e)) for e in data["errors"][:3])
        print(f"  GraphQL warnings: {msgs}", file=sys.stderr)
    entries = (data.get("data") or {}).get("entries") or []
    return [flatten_entry(e) for e in entries if e]


def chunks(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "pdb_id" not in reader.fieldnames:
            return set()
        for row in reader:
            pid = (row.get("pdb_id") or "").strip()
            if pid:
                done.add(pid.upper())
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("examples/data/pdb_validation.csv"),
        help="output CSV path (default: examples/data/pdb_validation.csv)",
    )
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="GraphQL batch size (max 1000)")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N IDs (for testing)")
    ap.add_argument(
        "--method",
        default=None,
        help="filter Search API by experimental_method (e.g. X-ray, EM, NMR)",
    )
    ap.add_argument(
        "--ids-from",
        type=Path,
        default=None,
        help="optional text/CSV file of PDB IDs (one per line, or pdb_id column)",
    )
    ap.add_argument(
        "--holdings",
        action="store_true",
        help="use holdings REST list instead of Search API (includes all current IDs)",
    )
    ap.add_argument("--sleep", type=float, default=0.05, help="pause between GraphQL batches (seconds)")
    ap.add_argument("--no-resume", action="store_true", help="overwrite output instead of appending")
    args = ap.parse_args()

    if not 1 <= args.batch_size <= 1000:
        ap.error("--batch-size must be in [1, 1000]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ids_cache = args.output.with_suffix(args.output.suffix + ".ids.txt")

    if args.ids_from is not None:
        ids = _read_id_file(args.ids_from)
        print(f"Loaded {len(ids):,} IDs from {args.ids_from}", file=sys.stderr)
        if args.limit is not None:
            ids = ids[: args.limit]
    elif args.holdings:
        ids = fetch_entry_ids_holdings()
        if args.limit is not None:
            ids = ids[: args.limit]
            print(f"Limiting to {len(ids):,} IDs", file=sys.stderr)
    else:
        # Stream IDs into cache as we page; incomplete caches are resumed.
        ids = fetch_entry_ids(
            method=args.method,
            limit=args.limit,
            cache_path=None if args.limit is not None else ids_cache,
        )
        if args.limit is None:
            print(f"ID cache ready: {ids_cache} ({len(ids):,})", file=sys.stderr)

    # Deduplicate while preserving order.
    seen: set = set()
    unique: List[str] = []
    for pid in ids:
        key = pid.strip().upper()
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
    ids = unique
    done = set() if args.no_resume else load_done(args.output)
    if done:
        before = len(ids)
        ids = [i for i in ids if i not in done]
        print(f"Resume: skipping {before - len(ids):,} already in {args.output}", file=sys.stderr)

    write_header = args.no_resume or not args.output.exists() or args.output.stat().st_size == 0
    mode = "w" if args.no_resume else "a"

    n_ok = 0
    n_fail = 0
    t0 = time.time()
    with args.output.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        total_batches = (len(ids) + args.batch_size - 1) // args.batch_size
        for bi, batch in enumerate(chunks(ids, args.batch_size), start=1):
            try:
                rows = fetch_batch(batch)
            except Exception as exc:  # noqa: BLE001 — keep going on a bad batch
                n_fail += len(batch)
                print(f"  batch {bi}/{total_batches} FAILED ({batch[0]}…): {exc}", file=sys.stderr)
                time.sleep(max(args.sleep, 2.0))
                continue

            got = {r["pdb_id"] for r in rows if r.get("pdb_id")}
            for pid in batch:
                if pid not in got:
                    # Still emit a stub so resume skips it; metrics stay blank.
                    rows.append({"pdb_id": pid})

            for row in rows:
                writer.writerow(row)
            f.flush()
            n_ok += len(got)
            n_fail += len(batch) - len(got)

            elapsed = time.time() - t0
            rate = n_ok / elapsed if elapsed > 0 else 0.0
            print(
                f"  batch {bi}/{total_batches}: +{len(got)} "
                f"(total {n_ok + len(done):,} @ {rate:.0f}/s)",
                file=sys.stderr,
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(
        f"Done. wrote={n_ok:,} missing={n_fail:,} → {args.output}",
        file=sys.stderr,
    )


def _read_id_file(path: Path) -> List[str]:
    text = path.read_text()
    if path.suffix.lower() == ".csv":
        import io

        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames and "pdb_id" in reader.fieldnames:
            return [row["pdb_id"].strip() for row in reader if row.get("pdb_id")]
        # Fall through: treat first column as IDs.
        reader = csv.reader(io.StringIO(text))
        return [row[0].strip() for row in reader if row and row[0].strip() and row[0].strip().lower() != "pdb_id"]
    return [line.strip().split(",")[0] for line in text.splitlines() if line.strip() and not line.startswith("#")]


if __name__ == "__main__":
    main()
