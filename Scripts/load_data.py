"""Load policy data from the SQLite knowledge base.

Reads the ``policy_data`` table from ``Data/policy_data.db`` and returns a
lightweight :class:`PolicyData` object whose API mirrors the subset of
``pandas.DataFrame`` that the downstream pipeline actually uses
(``len()``, ``iterrows()``), so ``chunk_data.py`` works with zero changes.
"""

import os
import sqlite3


class PolicyData:
    """Lightweight DataFrame-like wrapper around a list of policy rows.

    Supports the two operations the downstream pipeline requires:

    * ``len(data)`` — total number of rows (used by ``main.py``).
    * ``data.iterrows()`` — yields ``(index, row_dict)`` pairs where each
      ``row_dict`` supports ``row['ID']``, ``row['Topic']``, ``row['Content']``
      (used by ``chunk_data.py``).
    """

    def __init__(self, rows: list):
        self._rows = rows  # list of dicts with keys ID, Topic, Content

    def __len__(self) -> int:
        return len(self._rows)

    def iterrows(self):
        """Yield (index, row_dict) pairs, matching the pandas API."""
        for i, row in enumerate(self._rows):
            yield i, row


def load_db(filepath: str) -> PolicyData:
    """Load policy data from a SQLite database.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the ``.db`` file.

    Returns
    -------
    PolicyData
        A lightweight object with ``len()`` and ``iterrows()`` support,
        containing rows with keys ``ID``, ``Topic``, and ``Content``.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist on disk.
    ValueError
        If the database does not contain a ``policy_data`` table, or if
        the table is empty, or if the expected columns are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Database file not found: {filepath}")

    conn = sqlite3.connect(filepath)
    try:
        # Verify the table exists
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        if "policy_data" not in tables:
            raise ValueError(
                f"Database does not contain a 'policy_data' table. "
                f"Found tables: {tables}"
            )

        # Verify column names
        col_info = conn.execute("PRAGMA table_info(policy_data)").fetchall()
        actual_cols = {c[1] for c in col_info}
        expected_cols = {"ID", "Topic", "Content"}
        if not expected_cols.issubset(actual_cols):
            raise ValueError(
                f"Table policy_data is missing columns: "
                f"{expected_cols - actual_cols}"
            )

        # Fetch all rows as dicts
        cursor = conn.execute("SELECT ID, Topic, Content FROM policy_data")
        rows = [
            {"ID": r[0], "Topic": str(r[1]), "Content": str(r[2])}
            for r in cursor.fetchall()
        ]
    finally:
        conn.close()

    if not rows:
        raise ValueError("The policy_data table is empty.")

    return PolicyData(rows)
