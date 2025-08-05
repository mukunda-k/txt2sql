"""
Utilities for turning one or more CSV files into:
• Pandas dataframes
• A JSON schema with SQL data types
• A pretty printable schema
"""
from __future__ import annotations

from typing import List, Tuple, Dict
import pandas as pd
import json

# Pandas → SQL type mapping (feel free to extend)
_P2SQL = {
    "object": "VARCHAR",
    "int64": "INTEGER",
    "float64": "FLOAT",
    "bool": "BOOLEAN",
    "datetime64[ns]": "TIMESTAMP",
}


def _df_to_sql_types(df: pd.DataFrame) -> Dict[str, str]:
    return {col: _P2SQL.get(str(df[col].dtype), "VARCHAR") for col in df.columns}


def load_tables(paths: List[Tuple[str, str]]) -> Dict[str, pd.DataFrame]:
    """[(table_name, csv_path)] → {table_name: dataframe}"""
    tables: Dict[str, pd.DataFrame] = {}
    for table, path in paths:
        tables[table] = pd.read_csv(path)
    return tables


def json_schema(tables: Dict[str, pd.DataFrame]) -> str:
    """
    {"table": {"col": "TYPE", …}, …}
    Compact separators keep the prompt short.
    """
    schema = {t: _df_to_sql_types(df) for t, df in tables.items()}
    return json.dumps(schema, separators=(",", ":"))


def pretty_schema(tables: Dict[str, pd.DataFrame]) -> str:
    lines: List[str] = []
    for t, df in tables.items():
        lines.append(f"• {t}  ({len(df)} rows)")
        for c, typ in _df_to_sql_types(df).items():
            lines.append(f"    – {c}: {typ}")
    return "\n".join(lines)
