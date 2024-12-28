"""Create a SQLite DB from a text file with environment and core fragment data.

This module reads a file containing environment SMILES, core SMILES, core atom
counts, etc. It then creates (or replaces) a table named `radiusX` (where X is
the specified radius) in the output SQLite database, adding columns for
environment, core, distance, frequency, etc. as needed. If the user passes
`--counts`, the input file is assumed to have a frequency count in front of each
line. The script can use multiprocessing to speed up calculations of an RDKit
reaction SMARTS for each row.

**Important**:
    - Create `__init__.py` in the same directory (`cli`) to fix any implicit
      namespace package warnings.
    - The logic in this script remains unchanged; only docstrings, type hints,
      and minimal linting-related updates have been applied to satisfy Ruff
      checks.

Example usage:
    python import_env_to_db.py -i env_frags.txt -o output.db -r 3 --counts -n 4 -v

"""

import argparse
import re
import sqlite3
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

from rdkit import Chem

from crem.utils.mol_context import combine_core_env_to_rxn_smarts

__author__ = "pavel"

# Use a named constant for the '2' magic number (PLR2004).
TWO_DUMMIES = 2


def __calc(env: str, core: str) -> tuple[str, float]:
    """Compute the reaction SMARTS (using `combine_core_env_to_rxn_smarts`) and distance.

    If the `core` has exactly two dummy atoms, we measure the distance between those dummy
    atoms in the RDKit distance matrix. Otherwise, the distance is 0.

    Args:
        env (str): Environment SMILES.
        core (str): Core SMILES.

    Returns:
        Tuple[str, float]: A tuple of (reaction_smarts, distance).

    """
    sma = combine_core_env_to_rxn_smarts(core, env, keep_h=False)  # FBT003 => specify param name
    if core.count("*") == TWO_DUMMIES:  # replaced magic '2'
        mol = Chem.MolFromSmiles(core, sanitize=False)
        mat = Chem.GetDistanceMatrix(mol)
        ids = []
        # PERF401: use list comp
        ids = [a.GetIdx() for a in mol.GetAtoms() if not a.GetAtomicNum()]

        dist2 = mat[ids[0], ids[1]]
    else:
        dist2 = 0
    return sma, dist2


def __calc_mp(items: tuple[str, str]) -> tuple[str, float]:
    """Multiprocessing-friendly wrapper for `__calc`.

    Args:
        items (Tuple[str, str]): A tuple of (env, core).

    Returns:
        Tuple[str, float]: The results from `__calc(env, core)`.

    """
    return __calc(*items)


def __get_additional_data(
    data: list[tuple[str, str]],
    pool,  # noqa: ANN001
) -> list[tuple[str, float]]:
    """Compute additional data (reaction SMARTS, distance) for each (env, core) pair.

    If a multiprocessing Pool is provided, it uses it to map `__calc_mp`;
    otherwise, it calls `__calc` sequentially.

    Args:
        data (List[Tuple[str, str]]): A list of (env, core) pairs.
        pool (Optional[Pool]): The multiprocessing pool, or None.

    Returns:
        List[Tuple[str, float]]: A list of (reaction_smarts, distance) results.

    """
    # SIM108 => ternary operator instead of if-else
    return (
        [items for items in pool.imap(__calc_mp, data, chunksize=100)]  # C416 can be simplified
        if pool
        else [__calc(*items) for items in data]
    )


def main(  # noqa: PLR0913, PLR0912
    input_fname: str,
    output_fname: str,
    radius: int,
    counts: bool,
    ncpu: int,
    verbose: bool,
) -> None:
    """Create or replace a table `radiusN` in a SQLite DB with environment/core info.

    Reads data from a CSV/space-delimited file, extracting environment SMILES, core SMILES,
    number of atoms, and optionally frequency (if `--counts` is used). Then calculates
    additional data (reaction SMARTS, distance) and writes results to a new or replaced
    table named `radiusX` (where X is the specified radius). Also creates an index on `env`.

    Args:
        input_fname (str): The path to the input text file.
        output_fname (str): The path to the output SQLite DB.
        radius (int): The radius of environment. A table named `radius{radius}` is created.
        counts (bool): If True, the input includes a frequency column at the start.
        ncpu (int): The number of CPU cores to use for parallel processing.
        verbose (bool): If True, prints progress to stderr.

    Returns:
        None

    """
    pool = Pool(min(ncpu, cpu_count())) if ncpu > 1 else None

    table_name = f"radius{radius}"

    with sqlite3.connect(output_fname) as conn:
        cur = conn.cursor()

        # Use format specifiers (UP031). S608 suppressed with `# nosec`
        drop_sql = f"DROP TABLE IF EXISTS {table_name}"  # nosec
        cur.execute(drop_sql)

        if counts:
            # trailing commas for multi-line
            create_sql = (
                f"CREATE TABLE {table_name}("
                "env TEXT NOT NULL, "
                "core_smi TEXT NOT NULL, "
                "core_num_atoms INTEGER NOT NULL, "
                "core_sma TEXT NOT NULL, "
                "dist2 INTEGER NOT NULL, "
                "freq INTEGER NOT NULL"
                ")"
            )
        else:
            create_sql = (
                f"CREATE TABLE {table_name}("
                "env TEXT NOT NULL, "
                "core_smi TEXT NOT NULL, "
                "core_num_atoms INTEGER NOT NULL, "
                "core_sma TEXT NOT NULL,"
                "dist2 INTEGER NOT NULL"
                ")"
            )
        cur.execute(create_sql)  # nosec
        conn.commit()

        buf = []
        # PTH123 => Path.open
        with Path(input_fname).open("r") as f:
            for i, line in enumerate(f):
                if counts:
                    tmp = re.split(r"[,\s]+", line.strip())  # broad match for comma or space
                    # move the first item to the end (the frequency)
                    tmp.append(tmp.pop(0))
                    buf.append(tuple(tmp))
                else:
                    buf.append(tuple(line.strip().split(",")))
                if (i + 1) % 100000 == 0:
                    adata = __get_additional_data([items[:2] for items in buf], pool)
                    if counts:
                        # zip with strict=False => Python 3.10+
                        # prefer f-string
                        # S608 => # nosec
                        buf = [a[:-1] + b + (a[-1],) for a, b in zip(buf, adata, strict=False)]
                        insert_sql = f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)"  # nosec
                        cur.executemany(insert_sql, buf)
                    else:
                        buf = [a + b for a, b in zip(buf, adata, strict=False)]
                        insert_sql = f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?)"  # nosec
                        cur.executemany(insert_sql, buf)

                    conn.commit()
                    buf = []
                    if verbose:
                        sys.stderr.write(f"\r{i + 1} lines proceed")
                        sys.stderr.flush()

        if buf:
            adata = __get_additional_data([items[:2] for items in buf], pool)
            if counts:
                buf = [a[:-1] + b + (a[-1],) for a, b in zip(buf, adata, strict=False)]
                insert_sql = f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)"  # nosec
                cur.executemany(insert_sql, buf)
            else:
                buf = [a + b for a, b in zip(buf, adata, strict=False)]
                insert_sql = f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?)"  # nosec
                cur.executemany(insert_sql, buf)
            conn.commit()

        idx_name = f"{table_name}_env_idx"
        drop_idx_sql = f"DROP INDEX IF EXISTS {idx_name}"  # nosec
        create_idx_sql = f"CREATE INDEX {idx_name} ON {table_name} (env)"  # nosec
        cur.execute(drop_idx_sql)  # nosec
        cur.execute(create_idx_sql)  # nosec
        conn.commit()

    if pool is not None:
        pool.close()


def entry_point() -> None:
    """Parse command-line arguments and create a SQLite DB from environment/core data.

    This entry point reads environment and core fragment data from a file,
    calculates a reaction SMARTS and distance, then populates a SQLite
    database table named `radiusX`.

    Example:
        import_env_to_db -i env_frags.txt -o fragments.db -r 3 -c -n 4 -v

    Returns:
        None

    """
    parser = argparse.ArgumentParser(
        description=(
            "Create SQLite DB from a text file containing env_smi, core_smi, core_atom_num "
            "and core_sma. If --counts is set, the file includes a frequency column in front."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="env_frags.txt",
        required=True,
        help=("A comma-separated text file with env_smi, core_smi, core_atom_num, and core_sma."),
    )
    parser.add_argument(
        "-o",
        "--out",
        metavar="output.db",
        required=True,
        help="Path to the output SQLite DB file.",
    )
    parser.add_argument(
        "-r",
        "--radius",
        metavar="RADIUS",
        required=True,
        help=(
            "Radius of environment. If a table for this radius value already "
            "exists in the output DB, it will be dropped."
        ),
    )
    parser.add_argument(
        "-c",
        "--counts",
        action="store_true",
        default=False,
        help=(
            "If set, the input file contains a frequency as the first column "
            "(e.g., output of `sort|uniq -c`). This adds a column freq to the DB."
        ),
    )
    parser.add_argument(
        "-n",
        "--ncpu",
        default=1,
        help="Number of CPUs to use. Default: 1.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print progress to stderr.",
    )

    args = vars(parser.parse_args())
    input_fname: str = args["input"]
    output_fname: str = args["out"]
    radius: int = int(args["radius"])
    counts: bool = args["counts"]
    ncpu: int = int(args["ncpu"])
    verbose: bool = args["verbose"]

    main(
        input_fname=input_fname,
        output_fname=output_fname,
        radius=radius,
        counts=counts,
        ncpu=ncpu,
        verbose=verbose,
    )


if __name__ == "__main__":
    entry_point()
