"""Fragment input compounds by cutting bonds matching specific SMARTS patterns.

This module provides a command-line interface (CLI) to read SMILES from an
input file, optionally accompanied by an ID, and fragment them using RDKit
(`rdMMPA`), with options for heavy-atom-only or hydrogen-only splitting modes.
Results are written to an output file, which may include duplicated lines
that can be handled externally.

Note:
    - A file named `__init__.py` must exist in the same `cli` folder to avoid
      implicit namespace package warnings (INP001).
    - We have replaced magic numbers 2 and 60 with constants (`HYDROGEN_SPLIT_ATOMS` and
      `MAX_HYDROGENS_ALLOWED`) to address Ruff's suggestions.
    - We merged repeated equality checks into membership checks (`mode in {0, 1}`, etc.).
    - We switched from percent-formatting to f-strings, added trailing commas,
      and replaced nested `with` statements with a single context manager.
    - `# noqa` comments are used where we must suppress certain warnings
      (e.g., too many arguments in a function) without changing the logic.

"""

__author__ = "pavel"

import argparse
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMMPA

# Constants to replace magic values:
HYDROGEN_SPLIT_ATOMS = 2
MAX_HYDROGENS_ALLOWED = 60


def fragment_mol(
    smi: str,
    smi_id: str = "",
    mode: int = 0,
    sep_out: str = ",",
) -> set[str]:
    """Fragment a molecule based on a given mode, returning a set of output lines.

    Args:
        smi (str): The SMILES string for the molecule.
        smi_id (str): An optional ID for the molecule. Defaults to "".
        mode (int): Fragmentation mode.
            - 0: All atoms form a fragment.
            - 1: Heavy-atom only splitting.
            - 2: Hydrogen-atom only splitting.
            Defaults to 0.
        sep_out (str): The output separator for fields in the resulting lines.

    Returns:
        Set[str]: A set of lines with SMILES, ID, core, chains, and newline appended.

    """
    mol = Chem.MolFromSmiles(smi)
    outlines: set[str] = set()

    if mol is None:
        # Convert percent-format to f-string
        sys.stderr.write(f"Can't generate mol for: {smi}\n")
    else:
        # heavy atoms
        if mode in {0, 1}:  # PLR1714: merging (mode == 0) or (mode == 1)
            frags = rdMMPA.FragmentMol(
                mol,
                pattern="[!#1]!@!=!#[!#1]",
                maxCuts=4,
                resultsAsMols=False,
                maxCutBonds=30,
            )
            frags += rdMMPA.FragmentMol(
                mol,
                pattern="[!#1]!@!=!#[!#1]",
                maxCuts=3,
                resultsAsMols=False,
                maxCutBonds=30,
            )
            frags = set(frags)
            for core, chains in frags:
                # Add a trailing comma (COM812) after the last argument if multi-line
                output = sep_out.join((smi, smi_id, core, chains)) + "\n"
                outlines.add(output)

        # hydrogen splitting
        if mode in {1, 2}:  # merging (mode == 1) or (mode == 2)
            mol = Chem.AddHs(mol)
            n = mol.GetNumAtoms() - mol.GetNumHeavyAtoms()
            if n < MAX_HYDROGENS_ALLOWED:  # Replace magic value 60
                frags = rdMMPA.FragmentMol(
                    mol,
                    pattern="[#1]!@!=!#[!#1]",
                    maxCuts=1,
                    resultsAsMols=False,
                    maxCutBonds=100,  # trailing comma
                )
                for core, chains in frags:
                    output = sep_out.join((smi, smi_id, core, chains)) + "\n"
                    outlines.add(output)

    return outlines


def process_line(
    line: str,
    sep: str | None,
    mode: int,
    sep_out: str,
) -> set[str] | None:
    """Process a single line from the input file, returning a set of fragment lines or None.

    Args:
        line (str): A single line read from the input file.
        sep (Optional[str]): The input file separator (default is None, meaning Tab).
        mode (int): Fragmentation mode (0, 1, or 2).
        sep_out (str): The separator for the resulting output lines.

    Returns:
        Optional[Set[str]]: A set of lines containing fragment info, or None if line is empty.

    """
    tmp = line.strip().split(sep)
    if not tmp:
        return None
    if len(tmp) == 1:
        return fragment_mol(tmp[0], mode=mode, sep_out=sep_out)
    return fragment_mol(tmp[0], tmp[1], mode=mode, sep_out=sep_out)


def main(  # noqa: PLR0913
    input_fname: str,
    output_fname: str,
    mode: int,
    sep: str | None,
    ncpu: int,
    sep_out: str,
    verbose: bool,
) -> None:
    """Fragment an entire file of SMILES, writing results to an output file.

    Args:
        input_fname (str): Path to the input file (SMILES, optionally ID).
        output_fname (str): Path to the output file for storing fragments.
        mode (int): Fragmentation mode (0, 1, or 2).
        sep (Optional[str]): Separator in the input file. If None, defaults to Tab.
        ncpu (int): Number of CPU cores to use.
        sep_out (str): The output separator (default comma).
        verbose (bool): Whether to print progress to stderr.

    Returns:
        None

    """
    ncpu = min(cpu_count(), max(ncpu, 1))
    p = Pool(ncpu)

    # Merge nested with statements into one (SIM117) and use Path.open() (PTH123).
    with Path(output_fname).open("w") as out, Path(input_fname).open("r") as f:
        for i, res in enumerate(
            p.imap_unordered(
                partial(process_line, sep=sep, mode=mode, sep_out=sep_out),
                f,
                chunksize=100,
            ),
            1,
        ):
            if res:
                out.write("".join(res))

            if verbose and i % 1000 == 0:
                # Convert percent-format to f-string
                sys.stderr.write(f"\r{i} molecules fragmented")
                sys.stderr.flush()

    p.close()


def entry_point() -> None:
    """CLI entry point to parse arguments and execute `main` for fragmentation.

    This function sets up an argument parser for the fragmentation script,
    reads the user inputs, and calls `main()` with the parsed values.
    """
    parser = argparse.ArgumentParser(
        description="Fragment input compounds by cutting bonds matching bond SMARTS.",
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="input.smi",
        required=True,
        help="Input SMILES with optional comma-separated ID).",
    )
    parser.add_argument(
        "-o",
        "--out",
        metavar="output.txt",
        required=True,
        help="Fragmented molecules.",
    )
    parser.add_argument(
        "-s",
        "--sep",
        metavar="STRING",
        required=False,
        default=None,
        help="Separator in input file. Default: Tab.",
    )
    parser.add_argument(
        "-d",
        "--sep_out",
        metavar="STRING",
        required=False,
        default=",",
        help="Separator in the output file. Default: comma",
    )
    parser.add_argument(
        "-m",
        "--mode",
        metavar="INTEGER",
        required=False,
        default=0,
        choices=[0, 1, 2],
        type=int,
        help=(
            "Fragmentation mode: 0 - all atoms constitute a fragment, 1 - heavy atoms only, "
            "2 - hydrogen atoms only. Default: 0."
        ),
    )
    parser.add_argument(
        "-c",
        "--ncpu",
        metavar="NUMBER",
        required=False,
        default=1,
        help="Number of cpus used for computation. Default: 1.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print progress.",
    )

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "input":
            input_fname = v
        elif o == "out":
            output_fname = v
        elif o == "verbose":
            verbose = v
        elif o == "ncpu":
            ncpu = int(v)
        elif o == "sep":
            sep = v
        elif o == "sep_out":
            sep_out = v
        elif o == "mode":
            mode = v

    main(
        input_fname=input_fname,
        output_fname=output_fname,
        mode=mode,
        sep=sep,
        ncpu=ncpu,
        sep_out=sep_out,
        verbose=verbose,
    )


if __name__ == "__main__":
    entry_point()
