import csv
import os
import tempfile
import uuid
from typing import Dict, Optional, cast

import pandas as pd
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization


class WiomData:
    """Snowflake helper that mirrors the trimmed transaction handler."""

    def __init__(self, db_name: str) -> None:
        if db_name != "snowflake":
            raise ValueError("WiomData only supports 'snowflake' in this playground")

        self._creds_snowflake: Dict[str, Optional[str]] = {
            "user": "MAANAS",
            "account": "BSNUFYZ-IP42416",
            "private_key": "-----BEGIN ENCRYPTED PRIVATE KEY-----\nMIIFHTBXBgkqhkiG9w0BBQ0wSjApBgkqhkiG9w0BBQwwHAQI/YGrNmgzhTUCAggAMAwGCCqGSIb3DQIJBQAwHQYJYIZIAWUDBAEqBBBQkB4AlGEOSOiVNdj7gdKWBIIEwJcs7cu/QZ9O1piIkJqHxGI7LrcrgwkPEqmC3B6G0H5Ou6ORELynphb/wgY56MQuVKEUgQnB/lUTG/A+u49IWUS5saXiMWxierIDnuWMUd+2HmEYBTgtAm1NUCX0+kws vMbZado7z7xXnDivE4oPyazlTiEPCruJ23EDHHiR2+PqYJKfUXYtTJyuUhQias8k PYLFbrEHHVXox9IfVCpPwAZN+WZIvup1s0X+wJoSiQApKeSI8M6NcoDq1uKzEadX 2WYWUFtktGGxF+cZNHVbEWnIfAyCpCAe5NxZvQbEH7zdPibJA7l4lP+TxUGEZhWI bgOk3wDlpwTrd6ub5nSTkzNHtO3XnSvoEio+HhFbeA/jmmro6TA4HI5HyhsxuNcA MHq1oglIi6Auu9awYoJi1//2xm3dwBOvRwxVoMs3p60nMdkSQvThLAAqSvlD/TPD MHRUT4yMpOOKuDUPsPxvualFwUax5yFSp7Wo+3ytaZEUtogho3KDz+/F9ZxIVXRU W40tDJ3YHRcN31SSd+9oTvIuZl/7p7noPtCIyAFCS/NAaVq/J62KM7m2A8a0pRAp 2y6OQtn5TRUkKJZ6RJRegNAwJ1cV0CXite5tCZ5QhaVMMTr5fjkGsquDdgm950Vm Ig+Tx9fmH46VBuhM2lZ9lwE25ZGACnX9ZsjND/Cf6OYs3KDcGabfW5FEtq7JYVm5 H2BWWtqnPqKY27AItzXdkHc+stSmW3bghsLYpQ6CmPFqxjE14uzjBMqbwJ3dW7Nc LYBC/rfJU2UEA4DTu08yV9qMNNc2iUlYXm05GHHH2Z1O1MSE9YiFnjLROUMz8Faj TxV1xOLqadnaLgzC2U0+i2RAq5VCS0KnnwFmMVs60huBrIswWNDilRWJMBvXdPQ7 4s0zqXWMwGUQgNiPCBFRBy49/vVuo3xN8iAcCJRE5aKYewShCE91TNm32wjENV58 KYnyi6T5o4/xXaYIhRtw71ED6qIbgdEvPEnp7i1Td7oERYmliWYnNqPD9VYeBR3Q lJ2KBhIkgFo9btv6O/gILfK26mavPOO8Ne5PZ/I5+aWPa4hTZ/1GdxLlQDCkB3mY xHqB7593ObZddz0a83KFtr+G3w4JxUcFctUIlpez4ieUrKyBW0CAIkrchL3OWjtL pCZUBwS2QQwgbKo2kjZ2fBW9VjdyDvQLyUriXnM7u2Zn4K/b7IrRIN/8BbIo3v8i giSzRMbxauS8XMJM1mFTN13hAH19aF2JJfb6jVoARH6DAxppGDy7WMEnQRUWpSn4 06UgCE/RYKzhmdTqs4/fNaGRIS4lYuxSIEL0nL5LvjQJ5sFtFVJE778Ab7q3f1YJ hsJIQ4LH2VPq2hnoJXrAlpfLuWyu3lKk1BJRVF5w7jQVfsLNohjYjDbh1ZuVX1y+ 3Vl108FwoOo8YgdtX1pbDFaAjOKmOWLdhhujYM9GsXtMcOJpu0IEwJi4oq2JksSS 8RWD86EQY+lnIzksmfttAI+HX4KJ8HT+2z5fpdluRjjWbUH8adnFQ6zBSf8DtLX0 DK4KW50RJguI1JRA2R7uvIC6/qSJIlvFMqxFk6WwycyRqJFE+YHjW0p9GtX8LFGo hNqVn/w47BziKFq2+Xwtlqs=\n-----END ENCRYPTED PRIVATE KEY-----",
            "private_key_password": "django123",
            "warehouse": "DS_MED_WH",
            "database": "PROD_DB",
            "schema": "DS_TABLES",
        }

        self._connection_params = self._build_connection_params(self._creds_snowflake)

    def _build_connection_params(self, creds: Dict[str, Optional[str]]) -> Dict[str, object]:
        params: Dict[str, object] = {
            "user": creds["user"],
            "account": creds["account"],
            "warehouse": creds["warehouse"],
            "database": creds["database"],
            "schema": creds["schema"],
        }

        private_key_pem = creds.get("private_key")
        if private_key_pem:
            pk_password = creds.get("private_key_password")
            password_bytes = pk_password.encode() if pk_password else None
            p_key = serialization.load_pem_private_key(
                private_key_pem.encode(),
                password=password_bytes,
                backend=default_backend(),
            )
            params["private_key"] = p_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

        return params

    def _connect(self):
        return snowflake.connector.connect(**self._connection_params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def query(
        self, sql: str, cache_file: Optional[str] = None, cache_h: int = 1
    ) -> pd.DataFrame:
        sql_stripped = (sql or "").strip()
        if not sql_stripped:
            raise ValueError("Query cannot be empty")

        if cache_file and os.path.exists(cache_file):
            age_hours = (
                pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(cache_file), unit="s")
            ).total_seconds() / 3600.0
            if age_hours < cache_h:
                return cast(pd.DataFrame, pd.read_csv(cache_file))

        df = self.get_df(sql_stripped)

        if cache_file:
            df.to_csv(cache_file, index=False)

        return df

    def get_df(self, query: str) -> pd.DataFrame:
        print(f"[WiomData] snowflake_select_start: {query[:100]}")
        conn = self._connect()
        cur = conn.cursor()
        try:
            cur.execute(query)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
            return pd.DataFrame(rows, columns=pd.Index(cols))
        finally:
            cur.close()
            conn.close()

    def execute(self, query: str, commit: bool = True) -> None:
        print(f"[WiomData] snowflake_execute_start: {query[:100]}")
        conn = self._connect()
        cur = conn.cursor()
        try:
            cur.execute(query)
            if commit:
                conn.commit()
        except Exception as exc:  # pragma: no cover - defensive logging path
            conn.rollback()
            print(f"[WiomData] snowflake_execute_failed: {exc}")
            raise
        finally:
            cur.close()
            conn.close()

    def sync_df_to_table(
        self,
        *,
        df: pd.DataFrame,
        table_name: str,
        schema_dict: Dict[str, str],
        temp_table_name: Optional[str] = None,
    ) -> None:
        if df is None or df.empty:
            print(f"[WiomData] snowflake_sync_skipped_empty_df: {table_name}")
            return

        df = df.copy()
        df.columns = [c.upper() for c in df.columns]
        schema_dict = {k.upper(): v for k, v in schema_dict.items()}

        expected_cols = list(schema_dict.keys())
        if not set(expected_cols).issubset(df.columns):
            missing = sorted(set(expected_cols) - set(df.columns))
            raise ValueError(f"Schema mismatch. Missing columns: {missing}")

        df = cast(pd.DataFrame, df.loc[:, expected_cols])

        actual_table = table_name.upper()
        temp_table = (temp_table_name or f"TEMP_{actual_table}").upper()

        conn = self._connect()
        cur = conn.cursor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name
            df.to_csv(csv_path, index=False)

        staged_name = os.path.basename(csv_path)

        try:
            print(
                f"[WiomData] snowflake_sync_start: table={actual_table} rows={len(df)}"
            )

            cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
            cols_def = ", ".join([f'"{col}" {dtype}' for col, dtype in schema_dict.items()])
            cur.execute(f"CREATE TABLE {temp_table} ({cols_def})")

            cur.execute(
                f"PUT file://{csv_path} @~/{staged_name} AUTO_COMPRESS=FALSE"
            )
            cur.execute(
                f"COPY INTO {temp_table} FROM @~/{staged_name} "
                "FILE_FORMAT=(TYPE=CSV SKIP_HEADER=1 FIELD_OPTIONALLY_ENCLOSED_BY='\"' "
                "ERROR_ON_COLUMN_COUNT_MISMATCH=FALSE)"
            )
            cur.execute(f"REMOVE @~/{staged_name}")

            cur.execute(f"DROP TABLE IF EXISTS {actual_table}")
            cur.execute(f"ALTER TABLE {temp_table} RENAME TO {actual_table}")

            conn.commit()
            print(f"[WiomData] snowflake_sync_complete: table={actual_table}")
        except Exception as exc:  # pragma: no cover - defensive logging path
            conn.rollback()
            print(f"[WiomData] snowflake_sync_failed: table={actual_table} error={exc}")
            raise
        finally:
            try:
                os.remove(csv_path)
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            cur.close()
            conn.close()

    def merge_df_to_table(
        self,
        *,
        df: pd.DataFrame,
        table_name: str,
        schema_dict: Dict[str, str],
        key_columns: list[str],
        update_columns: Optional[list[str]] = None,
        temp_table_name: Optional[str] = None,
    ) -> None:
        if df is None or df.empty:
            print(f"[WiomData] snowflake_merge_skipped_empty_df: {table_name}")
            return

        df = df.copy()
        df.columns = [c.upper() for c in df.columns]
        schema_dict = {k.upper(): v for k, v in schema_dict.items()}
        expected_cols = list(schema_dict.keys())

        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Schema mismatch. Missing columns: {sorted(missing)}")

        key_columns = [c.upper() for c in key_columns]
        update_columns = [c.upper() for c in (update_columns or [])]
        if not update_columns:
            update_columns = [col for col in expected_cols if col not in key_columns]

        df = cast(pd.DataFrame, df.loc[:, expected_cols])

        actual_table = table_name.upper()
        temp_table = (
            temp_table_name or f"TEMP_{actual_table}_{uuid.uuid4().hex[:8]}"
        ).upper()

        conn = self._connect()
        cur = conn.cursor()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            csv_path = f.name
            writer = csv.writer(f)
            writer.writerow(expected_cols)
            writer.writerows(df.itertuples(index=False, name=None))

        staged_name = os.path.basename(csv_path)

        try:
            print(
                f"[WiomData] snowflake_merge_stage_start: table={actual_table} rows={len(df)}"
            )

            cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
            cols_def = ", ".join([f'"{col}" {dtype}' for col, dtype in schema_dict.items()])
            cur.execute(f"CREATE TABLE {temp_table} ({cols_def})")

            cur.execute(
                f"PUT file://{csv_path} @~/{staged_name} AUTO_COMPRESS=FALSE"
            )
            cur.execute(
                f"COPY INTO {temp_table} FROM @~/{staged_name} "
                "FILE_FORMAT=(TYPE=CSV SKIP_HEADER=1 FIELD_OPTIONALLY_ENCLOSED_BY='\"')"
            )
            cur.execute(f"REMOVE @~/{staged_name}")

            on_clause = " AND ".join([f"t.{col} = s.{col}" for col in key_columns])
            update_clause = ", ".join([f"{col} = s.{col}" for col in update_columns])
            insert_columns = ", ".join(expected_cols)
            insert_values = ", ".join([f"s.{col}" for col in expected_cols])

            merge_sql = f"""
                MERGE INTO {actual_table} AS t
                USING {temp_table} AS s
                ON {on_clause}
                WHEN MATCHED THEN UPDATE SET {update_clause}
                WHEN NOT MATCHED THEN INSERT ({insert_columns}) VALUES ({insert_values})
            """
            cur.execute(merge_sql)
            cur.execute(f"DROP TABLE IF EXISTS {temp_table}")

            conn.commit()
            print(f"[WiomData] snowflake_merge_complete: table={actual_table}")
        except Exception as exc:  # pragma: no cover
            conn.rollback()
            print(
                f"[WiomData] snowflake_merge_failed: table={actual_table} error={exc}"
            )
            raise
        finally:
            try:
                os.remove(csv_path)
            except Exception:  # pragma: no cover
                pass
            cur.close()
            conn.close()

    def sync_query_to_table(
        self,
        *,
        select_query: str,
        table_name: str,
        temp_table_name: Optional[str] = None,
    ) -> None:
        actual_table = table_name.upper()
        temp_table = (
            temp_table_name or f"TEMP_{actual_table}_{uuid.uuid4().hex[:8]}"
        ).upper()
        select_query_clean = select_query.strip().rstrip(";")

        conn = self._connect()
        cur = conn.cursor()
        try:
            print(
                f"[WiomData] snowflake_sync_query_start: table={actual_table} temp={temp_table}"
            )

            cur.execute(
                f"CREATE OR REPLACE TABLE {temp_table} AS ({select_query_clean})"
            )

            cur.execute(
                """
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
                  AND TABLE_NAME = %s
                """,
                (actual_table,),
            )
            result = cur.fetchone()
            target_exists = bool(result and result[0] > 0)

            if target_exists:
                cur.execute(f"ALTER TABLE {temp_table} SWAP WITH {actual_table}")
                cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
            else:
                cur.execute(f"ALTER TABLE {temp_table} RENAME TO {actual_table}")

            conn.commit()
            print(f"[WiomData] snowflake_sync_query_complete: table={actual_table}")
        except Exception as exc:  # pragma: no cover
            conn.rollback()
            print(
                f"[WiomData] snowflake_sync_query_failed: table={actual_table} temp={temp_table} error={exc}"
            )
            raise
        finally:
            cur.close()
            conn.close()

    # ------------------------------------------------------------------
    # Backwards compatibility helpers
    # ------------------------------------------------------------------
    def sync_df_to_snowflake(
        self, table_name: str, df: pd.DataFrame, schema_dict: Dict[str, str]
    ) -> None:
        self.sync_df_to_table(df=df, table_name=table_name, schema_dict=schema_dict)
