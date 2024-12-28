"""api.py.

A user-friendly API layer over the core CReM library.
This module exposes simpler function signatures for:

  - mutate()
  - grow()
  - link()

...all of which internally call the library's deeper-level functions:
  - mutate_mol
  - grow_mol
  - link_mols

Usage example:
    from crem.api import mutate, grow, link

    # mutate a molecule
    mutated_smiles_list = mutate("c1ccccc1C", "path/to/fragments.db", max_replacements=10)

    # grow a molecule
    grown_smiles_list = grow("CCO", "path/to/fragments.db", min_atoms=2, max_atoms=5)

    # link two molecules
    link_smiles_list = link("c1ccccc1", "NCC(=O)O", "path/to/fragments.db")

By default, each function returns a list of enumerated product SMILES.
We keep advanced usage open by letting extra kwargs pass through to the underlying calls.
"""

from __future__ import annotations

from rdkit import Chem

# Low-level CReM calls
from crem.core.operations import grow_mol, link_mols, mutate_mol


def mutate(
    mol: str | Chem.Mol,
    db_path: str,
    *,
    radius: int = 3,
    max_replacements: int | None = None,
    protect_atom_indices: list[int] | None = None,
    replace_atom_indices: list[int] | None = None,
    ring_only: bool = False,
    ncores: int = 1,
    **kwargs,
) -> list[str]:
    """Mutate a single molecule using the CReM fragment database at db_path.
    Returns a list of mutated product SMILES.

    Args:
        mol: Input molecule, given as either SMILES string or an RDKit Mol.
        db_path: Path to the CReM fragment database (SQLite .db).
        radius: Context radius for matched molecular pairs (default: 3).
        max_replacements: Limit how many enumerations are returned in total (None = no limit).
        protect_atom_indices: Indices in the molecule that must NOT be altered.
        replace_atom_indices: If set, only these indices are replaced (ignored if also in protect).
        ring_only: If True, attempt ring substitutions by enabling `replace_cycles=True`.
        ncores: Number of CPU cores to use internally for parallel enumeration (default: 1).
        **kwargs: Additional advanced arguments forwarded to `mutate_mol`.

    Returns:
        A list of product SMILES strings.

    Raises:
        ValueError: If the input `mol` SMILES is invalid or if the DB path is incorrect.

    """  # noqa: D205
    # Convert input SMILES -> RDKit Mol, if needed
    if isinstance(mol, str):
        rdk_mol = Chem.MolFromSmiles(mol)
        if rdk_mol is None:
            raise ValueError(f"Invalid SMILES string provided: '{mol}'")
    else:
        rdk_mol = mol

    # If user wants ring-only changes, set `replace_cycles` = True
    #   (this allows ignoring the size constraints on ring fragments).
    if ring_only:
        kwargs["replace_cycles"] = True

    # Now we call `mutate_mol`, which yields a generator. We convert to list of SMILES.
    products = mutate_mol(
        mol=rdk_mol,
        db_name=db_path,
        radius=radius,
        max_replacements=max_replacements,
        protected_ids=protect_atom_indices,
        replace_ids=replace_atom_indices,
        ncores=ncores,
        return_mol=False,  # we want SMILES only
        return_rxn=False,
        return_rxn_freq=False,
        **kwargs,
    )
    return list(products)


def grow(
    mol: str | Chem.Mol,
    db_path: str,
    *,
    radius: int = 3,
    min_atoms: int = 1,
    max_atoms: int = 2,
    max_replacements: int | None = None,
    protect_atom_indices: list[int] | None = None,
    ncores: int = 1,
    **kwargs,
) -> list[str]:
    """Grow a molecule by replacing hydrogens with new fragments from the DB.

    Args:
        mol: Input molecule as SMILES or RDKit Mol.
        db_path: Path to the fragment DB file (SQLite).
        radius: Context radius for environment matching (default: 3).
        min_atoms: Min number of new heavy atoms to add (default: 1).
        max_atoms: Max number of new heavy atoms to add (default: 2).
        max_replacements: Overall limit on how many enumerations to return.
        protect_atom_indices: Indices in the original molecule whose attached hydrogens
                              should be protected from replacement.
        ncores: Number of CPU cores for parallel enumeration.
        **kwargs: Additional advanced arguments for `grow_mol`.

    Returns:
        A list of product SMILES for grown derivatives.

    """
    if isinstance(mol, str):
        rdk_mol = Chem.MolFromSmiles(mol)
        if rdk_mol is None:
            raise ValueError(f"Invalid SMILES string provided: '{mol}'")
    else:
        rdk_mol = mol

    products = grow_mol(
        mol=rdk_mol,
        db_name=db_path,
        radius=radius,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        max_replacements=max_replacements,
        protected_ids=protect_atom_indices,
        ncores=ncores,
        return_mol=False,
        return_rxn=False,
        return_rxn_freq=False,
        **kwargs,
    )
    return list(products)


def link(
    mol1: str | Chem.Mol,
    mol2: str | Chem.Mol,
    db_path: str,
    *,
    radius: int = 3,
    distance: int | None = None,
    min_atoms: int = 1,
    max_atoms: int = 2,
    max_replacements: int | None = None,
    protect_atom_indices_1: list[int] | None = None,
    protect_atom_indices_2: list[int] | None = None,
    ncores: int = 1,
    **kwargs,
) -> list[str]:
    """Link two molecules by bridging them with an intermediate fragment from the DB.

    Args:
        mol1: First molecule as SMILES or RDKit Mol.
        mol2: Second molecule as SMILES or RDKit Mol.
        db_path: Path to the fragment DB.
        radius: Context radius for environment matching (default: 3).
        distance: Optional integer distance constraint for the bridging fragment.
        min_atoms: Minimum new heavy atoms in bridging piece (default: 1).
        max_atoms: Maximum new heavy atoms (default: 2).
        max_replacements: Limit on enumerations produced.
        protect_atom_indices_1: Atom indices in `mol1` to protect.
        protect_atom_indices_2: Atom indices in `mol2` to protect.
        ncores: CPU cores for parallel enumeration.
        **kwargs: Additional advanced arguments for `link_mols`.

    Returns:
        List of SMILES linking the two input molecules.

    Raises:
        ValueError: If an input SMILES is invalid.

    """

    def ensure_mol(x):
        if isinstance(x, str):
            tmp = Chem.MolFromSmiles(x)
            if tmp is None:
                raise ValueError(f"Invalid SMILES string provided: '{x}'")
            return tmp
        return x

    rdk1 = ensure_mol(mol1)
    rdk2 = ensure_mol(mol2)

    products = link_mols(
        mol1=rdk1,
        mol2=rdk2,
        db_name=db_path,
        radius=radius,
        dist=distance,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        max_replacements=max_replacements,
        protected_ids_1=protect_atom_indices_1,
        protected_ids_2=protect_atom_indices_2,
        ncores=ncores,
        return_mol=False,
        return_rxn=False,
        return_rxn_freq=False,
        **kwargs,
    )
    return list(products)
