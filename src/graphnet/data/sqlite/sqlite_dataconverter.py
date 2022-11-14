"""DataConverter for the SQLite backend."""

from collections import OrderedDict
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import sqlalchemy
import sqlite3
from tqdm import tqdm

from graphnet.data.dataconverter import DataConverter  # type: ignore[attr-defined]
from graphnet.data.sqlite.sqlite_utilities import run_sql_code, save_to_sql


class SQLiteDataConverter(DataConverter):
    """Class for converting I3-files to SQLite format."""

    # Class variables
    file_suffix = "db"

    # Abstract method implementation(s)
    def save_data(self, data: List[OrderedDict], output_file: str) -> None:
        """Save data to SQLite database."""
        # Check(s)
        if os.path.exists(output_file):
            self.warning(
                f"Output file {output_file} already exists. Appending."
            )

        # Concatenate data
        assert len(data)
        dataframe = OrderedDict([(key, pd.DataFrame()) for key in data[0]])
        for data_dict in data:
            for key, data_values in data_dict.items():
                df = construct_dataframe(data_values)

                if self.any_pulsemap_is_non_empty(data_dict) and len(df) > 0:
                    # only include data_dict in temp. databases if at least one pulsemap is non-empty,
                    # and the current extractor (df) is also non-empty (also since truth is always non-empty)
                    if len(dataframe[key]):
                        assert isinstance(dataframe[key], pd.DataFrame)
                        dataframe[key] = dataframe[key].append(
                            df, ignore_index=True, sort=True
                        )
                    else:
                        dataframe[key] = df

        # Save each dataframe to SQLite database
        self.debug(f"Saving to {output_file}")
        saved_any = False
        for table, df in dataframe.items():
            if len(df) > 0:
                save_to_sql(df, table, output_file)
                saved_any = True

        if saved_any:
            self.debug("- Done saving")
        else:
            self.warning(f"No data saved to {output_file}")

    def merge_files(
        self, output_file: str, input_files: Optional[List[str]] = None
    ) -> None:
        """SQLite-specific method for merging output files/databases.

        Args:
            output_file: Name of the output file containing the merged results.
            input_files: Intermediate files/databases to be merged, according
                to the specific implementation. Default to None, meaning that
                all files/databases output by the current instance are merged.
        """
        if input_files is None:
            self.info("Merging files output by current instance.")
            input_files = self._output_files

        if not output_file.endswith("." + self.file_suffix):
            output_file = ".".join([output_file, self.file_suffix])

        if os.path.exists(output_file):
            self.warning(
                f"Target path for merged database, {output_file}, already exists."
            )

        if len(input_files) > 0:
            self.info(f"Merging {len(input_files)} database files")

            # Create one empty database table for each extraction
            table_names = self._extract_table_names(input_files)
            for table_name in table_names:
                column_names = self._extract_column_names(
                    input_files, table_name
                )
                if len(column_names) > 1:
                    is_pulse_map = is_pulsemap_check(table_name)
                    self._create_table(
                        output_file,
                        table_name,
                        column_names,
                        is_pulse_map=is_pulse_map,
                    )

            # Merge temporary databases into newly created one
            self._merge_temporary_databases(output_file, input_files)
        else:
            self.warning("No temporary database files found!")

    # Internal methods
    def _extract_table_names(
        self, db: Union[str, List[str]]
    ) -> Tuple[str, ...]:
        """Get the names of all tables in database `db`."""
        if isinstance(db, list):
            results = [self._extract_table_names(path) for path in db]
            # @TODO: Check...
            assert all([results[0] == r for r in results])
            return results[0]

        with sqlite3.connect(db) as conn:
            table_names = tuple(
                [
                    p[0]
                    for p in (
                        conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table';"
                        ).fetchall()
                    )
                ]
            )

        return table_names

    def _extract_column_names(
        self, db_paths: List[str], table_name: str
    ) -> List[str]:
        for db_path in db_paths:
            with sqlite3.connect(db_path) as con:
                query = f"select * from {table_name} limit 1"
                columns = pd.read_sql(query, con).columns
            if len(columns):
                return columns
        return []

    def any_pulsemap_is_non_empty(self, data_dict: Dict[str, Dict]) -> bool:
        """Check whether there are non-empty pulsemaps extracted from P frame.

        Takes in the data extracted from the P frame, then retrieves the
        values, if there are any, from the pulsemap key(s) (e.g
        SplitInIcePulses). If at least one of the pulsemaps is non-empty then
        return true. If no pulsemaps exist, i.e., if no `I3FeatureExtractor` is
        called e.g. because `I3GenericExtractor` is used instead, always return
        True.
        """
        if len(self._pulsemaps) == 0:
            return True

        pulsemap_dicts = [data_dict[pulsemap] for pulsemap in self._pulsemaps]
        return any(d["dom_x"] for d in pulsemap_dicts)

    def _attach_index(self, database: str, table_name: str) -> None:
        """Attach the table index.

        Important for query times!
        """
        code = (
            "PRAGMA foreign_keys=off;\n"
            "BEGIN TRANSACTION;\n"
            f"CREATE INDEX event_no_{table_name} ON {table_name} (event_no);\n"
            "COMMIT TRANSACTION;\n"
            "PRAGMA foreign_keys=on;"
        )
        run_sql_code(database, code)

    def _create_table(
        self,
        database: str,
        table_name: str,
        columns: List[str],
        is_pulse_map: bool = False,
    ) -> None:
        """Create a table.

        Args:
            database: Path to the database.
            table_name: Name of the table.
            columns: The names of the columns of the table.
            is_pulse_map: Whether or not this is a pulse map table.
        """
        query_columns = list()
        for column in columns:
            if column == "event_no":
                if not is_pulse_map:
                    type_ = "INTEGER PRIMARY KEY NOT NULL"
                else:
                    type_ = "NOT NULL"
            else:
                type_ = "FLOAT"
            query_columns.append(f"{column} {type_}")
        query_columns_string = ", ".join(query_columns)

        code = (
            "PRAGMA foreign_keys=off;\n"
            f"CREATE TABLE {table_name} ({query_columns_string});\n"
            "PRAGMA foreign_keys=on;"
        )
        run_sql_code(database, code)

        if is_pulse_map:
            self.debug(table_name)
            self.debug("Attaching indices")
            self._attach_index(database, table_name)
        return

    def _submit_to_database(
        self, database: str, key: str, data: pd.DataFrame
    ) -> None:
        """Submit data to the database with specified key."""
        if len(data) == 0:
            self.info(f"No data provided for {key}.")
            return
        engine = sqlalchemy.create_engine("sqlite:///" + database)
        data.to_sql(key, engine, index=False, if_exists="append")
        engine.dispose()

    def _extract_everything(self, db: str) -> "OrderedDict[str, pd.DataFrame]":
        """Extract everything from the temporary database `db`.

        Args:
            db: Path to temporary database.

        Returns:
            Dictionary containing the data for each extracted table.
        """
        results = OrderedDict()
        table_names = self._extract_table_names(db)
        with sqlite3.connect(db) as conn:
            for table_name in table_names:
                query = f"select * from {table_name}"
                try:
                    data = pd.read_sql(query, conn)
                except:  # noqa: E722
                    data = []
                results[table_name] = data
        return results

    def _merge_temporary_databases(
        self,
        output_file: str,
        input_files: List[str],
    ) -> None:
        """Merge the temporary databases.

        Args:
            output_file: path to the final database
            input_files: list of names of temporary databases
        """
        for input_file in tqdm(input_files, colour="green"):
            results = self._extract_everything(input_file)
            for table_name, data in results.items():
                self._submit_to_database(output_file, table_name, data)


# Implementation-specific utility function(s)
def construct_dataframe(extraction: Dict[str, Any]) -> pd.DataFrame:
    """Convert extraction to pandas.DataFrame.

    Args:
        extraction: Dictionary with the extracted data.

    Returns:
        Extraction as pandas.DataFrame.
    """
    all_scalars = True
    for value in extraction.values():
        if isinstance(value, (list, tuple, dict)):
            all_scalars = False
            break

    out = pd.DataFrame(extraction, index=[0] if all_scalars else None)
    return out


def is_pulsemap_check(table_name: str) -> bool:
    """Check whether `table_name` corresponds to a pulsemap."""
    if "pulse" in table_name.lower():
        return True
    else:
        return False
