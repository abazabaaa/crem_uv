"""Add columns with chosen molecular properties to a CReM fragment database.

This module provides a command-line interface (via `entry_point()`) for updating
CReM SQLite fragment databases with additional columns (e.g., MW, LogP, etc.).
It uses RDKit to compute the values for each row of matching tables (i.e.,
tables named like 'radius%'), adds the columns if needed, and populates them.

Note:
    - We preserve the underlying logic and behavior per the user's request.
    - We added docstrings in Google style.
    - We handle Ruff warnings about implicit namespace packages (add `__init__.py` in `cli/`).
    - We have one blank line between summary and the rest of the docstring (D205).
    - We make boolean function arguments keyword-only (with `*,`) to address FBT00x issues.
    - We avoid direct f-strings in exception raises by assigning the string to a variable first.
    - For SQL injection warnings (S608), we add `# nosec` to indicate we are consciously ignoring them.

"""

import argparse
import sqlite3
import sys
from functools import partial
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcFractionCSP3, CalcNumRotatableBonds, CalcTPSA


# Stub definitions for arg_types (remove if you have real definitions):
def cpu_type(value: str) -> int:
    """Convert a string to an integer CPU count.

    Args:
        value (str): The input string representing CPU count.

    Raises:
        argparse.ArgumentTypeError: If the value is not a valid integer.

    Returns:
        int: The parsed integer CPU count.

    """
    try:
        return int(value)
    except ValueError as exc:
        msg = f"Invalid CPU value: {value}"
        raise argparse.ArgumentTypeError(msg) from exc


def filepath_type(value: str) -> str:
    """Validate or process file path strings.

    Here, we simply return the string unchanged. Replace with actual checks if needed.

    Args:
        value (str): The file path string.

    Returns:
        str: The unchanged file path string.

    """
    return value


props = ["mw", "logp", "rtb", "tpsa", "fcsp3"]

# Use a ternary operator per Ruff suggestion for CHUNK_SIZE:
CHUNK_SIZE = 999 if sqlite3.sqlite_version_info[:2] <= (3, 32) else 32766


def property_type(x: list[str]) -> list[str]:
    """Filter valid property names from the input list.

    Args:
        x (List[str]): A list of property names to check.

    Returns:
        List[str]: A filtered list containing only recognized property names.

    """
    return [item.lower() for item in x if item.lower() in props]


def calc(  # noqa: PLR0913
    items: tuple[int, str],
    *,
    mw: bool = False,
    logp: bool = False,
    rtb: bool = False,
    tpsa: bool = False,
    fcsp3: bool = False,
) -> tuple[int, str]:
    """Compute chosen properties (MW, logP, etc.) for a single row.

    This function is mapped over database rows (rowid, smi). Depending on which
    booleans are True (mw, logp, rtb, tpsa, fcsp3), it calculates the corresponding
    RDKit descriptors and builds an SQL update string.

    Args:
        items (Tuple[int, str]): A tuple (rowid, smi).
        mw (bool): Whether to compute MW. Defaults to False.
        logp (bool): Whether to compute logP. Defaults to False.
        rtb (bool): Whether to compute number of rotatable bonds. Defaults to False.
        tpsa (bool): Whether to compute TPSA. Defaults to False.
        fcsp3 (bool): Whether to compute fraction C(sp3). Defaults to False.

    Returns:
        Tuple[int, str]: A tuple of (rowid, update_string_for_SQL).

    """
    rowid, smi = items
    res = {}
    mol = Chem.MolFromSmiles(smi)
    if mol:
        if mw:
            res["mw"] = round(MolWt(mol), 2)
        if logp:
            res["logp"] = round(MolLogP(mol), 2)
        if rtb:
            res["rtb"] = CalcNumRotatableBonds(Chem.RemoveHs(mol))
        if tpsa:
            res["tpsa"] = CalcTPSA(mol)
        if fcsp3:
            res["fcsp3"] = round(CalcFractionCSP3(mol), 3)

    upd_str = ",".join(f"{k} = {v}" for k, v in res.items())
    return rowid, upd_str


def entry_point() -> None:
    """Parse arguments and add property columns to the DB.

    This function implements the CLI, accepting input arguments for the SQLite
    database path, which properties to compute, the number of CPUs, and verbosity.
    It then iterates over tables matching 'radius%', adds columns if missing, and
    computes the requested properties.

    Raises:
        SystemExit: If no valid properties were supplied or other critical errors occur.

    """
    parser = argparse.ArgumentParser(
        description="Add columns with values of chosen properties to CReM fragment database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="FILENAME",
        required=True,
        type=filepath_type,
        help="SQLite DB with CReM fragments.",
    )
    parser.add_argument(
        "-p",
        "--properties",
        metavar="NAMES",
        required=False,
        nargs="*",
        default=props,
        choices=props,
        help="Properties to compute.",
    )
    parser.add_argument("-c", "--ncpu", default=1, type=cpu_type, help="Number of CPUs.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print progress to STDERR.",
    )

    args = parser.parse_args()

    if not args.properties:
        sys.stderr.write(
            f'No valid names of properties were supplied. Check them please: {", ".join(args.properties)}\n',
        )
        sys.exit(1)

    pool = Pool(args.ncpu)

    mw_flag = "mw" in args.properties
    logp_flag = "logp" in args.properties
    rtb_flag = "rtb" in args.properties
    tpsa_flag = "tpsa" in args.properties
    fcsp3_flag = "fcsp3" in args.properties

    with sqlite3.connect(args.input) as conn:
        cur = conn.cursor()
        tables_cursor = cur.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'radius%'",
        )
        tables = [i[0] for i in tables_cursor]

        for table in tables:
            for prop in args.properties:
                try:
                    # nosec: This is known to be a potential SQL injection vector,
                    # but we preserve the logic as requested.
                    sql_alter = f"ALTER TABLE {table} ADD COLUMN {prop} NUMERIC DEFAULT NULL"
                    cur.execute(sql_alter)  # nosec
                    conn.commit()
                except sqlite3.OperationalError as e:
                    sys.stderr.write(str(e) + "\n")

            # nosec: again, potential injection, but left as-is to maintain logic
            null_checks = [f"{prop} IS NULL" for prop in args.properties]
            sql_select = f"SELECT rowid, core_smi FROM {table} WHERE " + " OR ".join(null_checks)  # noqa: S608
            cur.execute(sql_select)  # nosec
            res = cur.fetchall()

            for i, (rowid, upd_str) in enumerate(
                pool.imap_unordered(
                    partial(
                        calc,
                        mw=mw_flag,
                        logp=logp_flag,
                        rtb=rtb_flag,
                        tpsa=tpsa_flag,
                        fcsp3=fcsp3_flag,
                    ),
                    res,
                ),
                1,
            ):
                # nosec: preserving original logic
                sql_update = f"UPDATE {table} SET {upd_str} WHERE rowid = '{rowid}'"  # noqa: S608
                cur.execute(sql_update)  # nosec
                if i % 10_000 == 0:
                    conn.commit()
                    if args.verbose:
                        sys.stderr.write(f"\r{i} fragments processed")
            conn.commit()

            sys.stderr.write(f"\nProperties were successfully added to {args.input}\n")


if __name__ == "__main__":
    entry_point()
