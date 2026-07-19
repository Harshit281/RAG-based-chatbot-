"""
prepare_opp115.py — Fetch the OPP-115 privacy-policy dataset from Hugging Face
and fully replace the policy_data table in Data/policy_data.db.

Usage:
    python Scripts/prepare_opp115.py

Prerequisites:
    pip install datasets

Output:
    Data/policy_data.db  (SQLite, table: policy_data with columns ID, Topic, Content)
"""

import os
import re
import sqlite3

# -- 1. Install check ---------------------------------------------------------
try:
    from datasets import load_dataset  # type: ignore
except ImportError:
    raise SystemExit("Run: pip install datasets")

# -- Integer label -> OPP-115 raw category name -------------------------------
LABEL_TO_RAW_CATEGORY = {
    0:  "Data Retention",
    1:  "Data Security",
    2:  "Do Not Track",
    3:  "First Party Collection/Use",
    4:  "International and Specific Audiences",
    5:  "Other",
    6:  "Policy Change",
    7:  "Third Party Sharing/Collection",
    8:  "User Access, Edit and Deletion",
    9:  "User Choice/Control",
    10: "Introductory/Generic",
    11: "Practice Not Covered",
}

# -- Category mapping (raw -> cleaned display name) ----------------------------
CATEGORY_MAP = {
    "First Party Collection/Use":           "First Party Collection and Use",
    "Third Party Sharing/Collection":       "Third Party Sharing and Collection",
    "User Choice/Control":                  "User Choice and Control",
    "User Access, Edit and Deletion":       "User Access Edit and Deletion",
    "Data Retention":                       "Data Retention",
    "Data Security":                        "Data Security",
    "Policy Change":                        "Policy Change",
    "Do Not Track":                         "Do Not Track",
    "International and Specific Audiences": "International and Specific Audiences",
    "Other":                                "Other",
}


def _resolve_topic(raw_label) -> str:
    """Convert a label (int, list-of-ints, or string) to a display topic name."""
    # Handle list of labels — use the first one as the primary topic
    if isinstance(raw_label, (list, tuple)):
        raw_label = raw_label[0] if raw_label else 5  # default to "Other" (5)

    if isinstance(raw_label, int):
        raw_cat = LABEL_TO_RAW_CATEGORY.get(raw_label, "Other")
    else:
        raw_cat = str(raw_label)

    return CATEGORY_MAP.get(raw_cat, raw_cat)


def main() -> None:
    # -- 2. Connect & reset table ----------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, os.pardir, "Data")
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, "policy_data.db")
    db_path = os.path.normpath(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS policy_data")
        cur.execute(
            """
            CREATE TABLE policy_data (
                ID      TEXT PRIMARY KEY,
                Topic   TEXT NOT NULL,
                Content TEXT NOT NULL
            )
            """
        )
        conn.commit()
        print("[0/5] Old policy_data table dropped and recreated.")

        # -- 3. Load dataset ---------------------------------------------------
        ds = load_dataset("alzoubi36/opp_115", split="train")
        print(f"[1/5] Loaded {len(ds)} segments from OPP-115.")
        col_names = list(ds.column_names)
        print(f"      Available columns: {col_names}")

        # -- 4. Column detection (dynamic) -------------------------------------
        TEXT_COL = None
        CAT_COL = None
        ID_COL = None

        for c in col_names:
            if "text" in c.lower() and TEXT_COL is None:
                TEXT_COL = c
            if ("categor" in c.lower() or "label" in c.lower()) and CAT_COL is None:
                CAT_COL = c

        for candidate in ("id", "segment_id", "policy_id"):
            if candidate in col_names:
                ID_COL = candidate
                break

        if TEXT_COL is None:
            raise SystemExit(
                f"Cannot find a text column. Available columns: {col_names}\n"
                "Expected a column whose name contains 'text'."
            )

        # -- 5 & 6. Build rows -------------------------------------------------
        rows: list[dict[str, str]] = []
        seen_ids: dict[str, int] = {}

        for i, item in enumerate(ds):
            # --- Content ---
            raw_text = str(item.get(TEXT_COL, ""))
            content = re.sub(r"\s+", " ", raw_text).strip()
            if not content:
                continue

            # --- Topic ---
            if CAT_COL is not None:
                raw_label = item.get(CAT_COL, "Other")
                topic = _resolve_topic(raw_label)
            else:
                topic = "Other"

            # --- ID (with duplicate handling) ---
            if ID_COL is not None:
                base_id = f"OPP-{item[ID_COL]}"
            else:
                base_id = f"OPP-{i + 1:04d}"

            if base_id in seen_ids:
                seen_ids[base_id] += 1
                row_id = f"{base_id}-{seen_ids[base_id]}"
            else:
                seen_ids[base_id] = 0
                row_id = base_id

            rows.append({"ID": row_id, "Topic": topic, "Content": content})

        print(f"[2/5] Built {len(rows)} rows.")

        # -- 7. Validate -------------------------------------------------------
        if len(rows) == 0:
            raise SystemExit("No rows built. Check dataset columns.")

        expected_keys = {"ID", "Topic", "Content"}
        for r in rows:
            assert set(r.keys()) == expected_keys, f"Bad keys: {set(r.keys())}"
            assert r["Content"], f"Empty Content for ID={r['ID']}"

        all_ids = [r["ID"] for r in rows]
        assert len(all_ids) == len(set(all_ids)), "Duplicate IDs remain after dedup!"
        print("[3/5] Validation passed.")

        # -- 8. Insert (single transaction) ------------------------------------
        cur.execute("BEGIN")
        try:
            cur.executemany(
                "INSERT INTO policy_data (ID, Topic, Content) VALUES (?, ?, ?)",
                [(r["ID"], r["Topic"], r["Content"]) for r in rows],
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        print(f"[4/5] Inserted {len(rows)} rows into {os.path.abspath(db_path)}")

        # -- 9. Verify ---------------------------------------------------------
        count = cur.execute("SELECT COUNT(*) FROM policy_data").fetchone()[0]
        col_info = cur.execute("PRAGMA table_info(policy_data)").fetchall()
        actual_cols = [c[1] for c in col_info]
        assert actual_cols == ["ID", "Topic", "Content"], (
            f"Column mismatch: {actual_cols}"
        )
        assert count == len(rows), (
            f"Row count mismatch: DB has {count}, expected {len(rows)}"
        )
        print(f"[5/5] Verified. Table has {count} rows and 3 columns.")

        # -- 10. Summary -------------------------------------------------------
        print("\n-- Topic Distribution " + "-" * 35)
        dist = cur.execute(
            "SELECT Topic, COUNT(*) as count "
            "FROM policy_data GROUP BY Topic ORDER BY count DESC"
        ).fetchall()
        print(f"  {'Topic':<45} {'Count':>6}")
        print(f"  {'-' * 45} {'-' * 6}")
        for topic, cnt in dist:
            print(f"  {topic:<45} {cnt:>6}")

    finally:
        # -- 11. Close ---------------------------------------------------------
        conn.close()

    # -- 12. Remind user -------------------------------------------------------
    print(
        """
All done! Old policy_data table has been replaced with OPP-115 data.

Next steps:
  1. Delete cache:
       Mac/Linux -> rm -rf Data/cache
       Windows   -> rmdir /s /q Data\\cache
  2. Run chatbot: python main.py
"""
    )


if __name__ == "__main__":
    main()
