"""Create a text file for fragment replacement from fragmented molecules.

This module reads a fragment file (e.g., from `fragmentation.py`), processes each
line to extract environment and core fragments, optionally uses a user-specified
list of molecules to keep, and writes out a text file for downstream usage. It
can handle multiple CPU cores via multiprocessing and filter results based on
the max number of heavy atoms. Duplicated lines should be filtered externally.

Note:
    - We keep the original logic, using global variables, etc.
    - We silence complexity or global usage warnings where needed.
    - We provide Google-style docstrings and type hints.

"""

__author__ = "pavel"

import argparse
import sys
from itertools import permutations
from multiprocessing import Pool, cpu_count
from pathlib import Path

from rdkit import Chem

from crem.utils.mol_context import get_std_context_core_permutations

# Instead of magic value "2", we define a constant for clarity.
CUT_FRAG_COUNT = 2

# Global placeholders for state set in `init()`.
_keep_mols = set()
_radius = 1
_keep_stereo = False
_max_heavy_atoms = 20
_store_comp_id = False
_sep = ","


def process_line(line: str) -> list[tuple[str, ...]]:  # noqa: C901, PLR0912
    """Process a single line of fragment data.

    This function parses the input line (SMILES, ID, core, context), applies
    checks for missing fragments or user filters, and returns tuples of
    environment, core, heavy-atom count, plus (optionally) compound ID.

    Args:
        line (str): A line of text containing SMILES, ID, core, and context, separated by `_sep`.

    Returns:
        List[Tuple[str, ...]]: A list of tuples describing extracted fragments.

    """
    output = []
    smi, raw_id, core, context = line.strip().split(_sep)  # shadowing "id" -> rename to raw_id

    if (not core and not context) or (_keep_mols and raw_id not in _keep_mols):
        return output

    # one split
    if not core:
        residues = context.split(".")
        if len(residues) == CUT_FRAG_COUNT:  # using our constant
            for ctx_part, cr_part in permutations(residues, 2):
                if ctx_part == "[H][*:1]":  # ignore such cases
                    continue
                mm = Chem.MolFromSmiles(cr_part, sanitize=False)
                num_heavy_atoms = mm.GetNumHeavyAtoms() if mm else float("inf")
                if num_heavy_atoms <= _max_heavy_atoms:
                    env, cores = get_std_context_core_permutations(
                        ctx_part,
                        cr_part,
                        _radius,
                        _keep_stereo,
                    )
                    if env and cores:
                        if not _store_comp_id:
                            output.append((env, cores[0], str(num_heavy_atoms)))
                        else:
                            output.append((env, cores[0], str(num_heavy_atoms), raw_id))
        else:
            # changed from percent-format to f-string
            sys.stderr.write(f"more than two fragments in context ({context}) where core is empty\n")
            sys.stderr.flush()
    # two or more splits
    else:
        mm = Chem.MolFromSmiles(core, sanitize=False)
        num_heavy_atoms = mm.GetNumHeavyAtoms() if mm else float("inf")
        if num_heavy_atoms <= _max_heavy_atoms:
            env, cores = get_std_context_core_permutations(context, core, _radius, _keep_stereo)
            if env and cores:
                for c in cores:
                    if not _store_comp_id:
                        output.append((env, c, str(num_heavy_atoms)))
                    else:
                        output.append((env, c, str(num_heavy_atoms), raw_id))
    return output


def init(  # noqa: PLR0913
    keep_mols: str | None,
    radius: int,
    keep_stereo: bool,
    max_heavy_atoms: int,
    store_comp_id: bool,
    sep: str,
) -> None:
    """Initialize global variables for the processing.

    Args:
        keep_mols (Optional[str]): Path to a file with molecule IDs to keep.
        radius (int): Radius of molecular context.
        keep_stereo (bool): Whether to keep stereochemistry.
        max_heavy_atoms (int): Maximum number of heavy atoms in cores.
        store_comp_id (bool): If True, store compound ID in the output.
        sep (str): The delimiter for parsing lines.

    Note:
        This function uses global variables to store state, which is generally
        discouraged but preserved here to retain original logic. We silence
        the warnings using noqa.

    """
    # Using the global statement is discouraged, but we keep it for minimal code change.
    global _keep_mols  # noqa: PLW0603
    global _radius  # noqa: PLW0603
    global _keep_stereo  # noqa: PLW0603
    global _max_heavy_atoms  # noqa: PLW0603
    global _store_comp_id  # noqa: PLW0603
    global _sep  # noqa: PLW0603

    if keep_mols:
        # Use a set comprehension and context manager. Also `Path.open()` per ruff suggestions.
        with Path(keep_mols).open("r") as file_in:
            _keep_mols = {ln.strip() for ln in file_in}
    else:
        _keep_mols = set()

    _radius = radius
    _keep_stereo = keep_stereo
    _max_heavy_atoms = max_heavy_atoms
    _store_comp_id = store_comp_id
    _sep = sep


def main(  # noqa: PLR0913
    input_fname: str,
    output_fname: str,
    keep_mols: str | None,
    radius: int,
    keep_stereo: bool,
    max_heavy_atoms: int,
    ncpu: int,
    store_comp_id: bool,
    sep: str,
    verbose: bool,
) -> None:
    """Process the input fragment file and produce a text output file.

    Args:
        input_fname (str): Path to the input file with fragmented molecules.
        output_fname (str): Path to the output text file.
        keep_mols (Optional[str]): File with molecule names to keep, or None.
        radius (int): Radius of molecular context (in bonds).
        keep_stereo (bool): Whether to keep stereochemistry.
        max_heavy_atoms (int): Maximum number of heavy atoms in cores.
        ncpu (int): Number of CPU cores to use.
        store_comp_id (bool): If True, stores the compound ID in the output.
        sep (str): Separator in the input file.
        verbose (bool): Whether to print progress to stderr.

    Returns:
        None

    """
    ncpu = min(cpu_count(), max(ncpu, 1))
    p = Pool(ncpu, initializer=init, initargs=(keep_mols, radius, keep_stereo, max_heavy_atoms, store_comp_id, sep))

    try:
        # Single context manager for writing & reading
        with Path(output_fname).open("w") as out, Path(input_fname).open("r") as f:
            for i, res in enumerate(p.imap_unordered(process_line, f, chunksize=1000), start=1):
                for item in res:
                    if item:
                        out.write(",".join(item) + "\n")

                if verbose and i % 1000 == 0:
                    sys.stderr.write(f"\r{i} lines passed")
                    sys.stderr.flush()

    finally:
        p.close()


def entry_point() -> None:
    """Entry point for the command-line interface (CLI).

    This function parses command-line arguments, then calls `main()` to
    create a text file for fragment replacement from fragmented molecules.
    The output may contain duplicates, which should be filtered externally.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create text file for fragment replacement from fragmented molecules "
            "obtained with fragmentation.py. The output may contain duplicated "
            "lines which should be filtered out externally."
        ),
    )
    parser.add_argument("-i", "--input", metavar="frags.txt", required=True, help="Fragmented molecules.")
    parser.add_argument("-o", "--out", metavar="output.txt", required=True, help="Output text file.")
    parser.add_argument(
        "-d",
        "--sep",
        metavar="STRING",
        default=",",
        help="Separator/delimiter in the input file. Default: comma",
    )
    parser.add_argument(
        "-k",
        "--keep_mols",
        metavar="molnames.txt",
        default=None,
        help="File with mol names to keep. Others are ignored if not listed.",
    )
    parser.add_argument(
        "-r",
        "--radius",
        metavar="NUMBER",
        default=1,
        help="Radius of molecular context (in bonds). Default: 1.",
    )
    parser.add_argument(
        "-a",
        "--max_heavy_atoms",
        metavar="NUMBER",
        default=20,
        help="Maximum number of heavy atoms in cores. Default: 20.",
    )
    parser.add_argument(
        "-s",
        "--keep_stereo",
        action="store_true",
        default=False,
        help="Keep stereo in context and core parts.",
    )
    parser.add_argument(
        "-c",
        "--ncpu",
        metavar="NUMBER",
        default=1,
        help="Number of cpus used for computation. Default: 1.",
    )
    parser.add_argument(
        "--store_comp_id",
        action="store_true",
        default=False,
        help="Store compound ID in output (only for debug).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Print progress.")

    args = parser.parse_args()

    main(
        input_fname=args.input,
        output_fname=args.out,
        keep_mols=args.keep_mols,
        radius=int(args.radius),
        keep_stereo=args.keep_stereo,
        max_heavy_atoms=int(args.max_heavy_atoms),
        ncpu=int(args.ncpu),
        store_comp_id=args.store_comp_id,
        sep=args.sep,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    entry_point()
