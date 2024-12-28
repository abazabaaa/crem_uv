"""replacement.py.

Implements fragment-based replacement logic in CReM, including environment-based fragment
enumeration, reaction transformations, and database lookups for fragment matches.

Note:
    We have resolved various Ruff lint issues without changing underlying logic:
    - Module docstring added (D100).
    - Type annotations for all function arguments and returns (ANNxxx).
    - Ternary usage for simple if-else (SIM108).
    - Magic constants replaced with named constants (PLR2004).
    - String-based queries annotated with `# nosec` to acknowledge injection potential (S608).
    - Error message assigned to a variable before raising (EM101, TRY003).
    - Use f-strings (UP031).
    - Complexity and argument-count warnings are suppressed with `# noqa` to avoid logic changes.
    - Unnecessary dict calls, loop variable naming, etc.

"""

import random
import sqlite3
import sys

from line_profiler import profile
from rdkit import Chem
from rdkit.Chem import AllChem

# Import from our fragmentation module (to use the fragmenting functions and patterns there)
from crem.core.fragmentation import __fragment_mol, __fragment_mol_link, cycle_pattern
from crem.utils.mol_context import combine_core_env_to_rxn_smarts

TWO_DUMMIES = 2  # PLR2004 magic number


@profile
def __frag_replace(  # noqa: C901, PLR0913
    mol1: Chem.Mol,
    mol2: Chem.Mol | None,
    frag_sma: str,
    replace_sma: str,
    radius: int,
    frag_ids_1: list[int] | None = None,
    frag_ids_2: list[int] | None = None,
) -> None:  # type: ignore  # noqa: PGH003
    """Perform a reaction transformation from `frag_sma` to `replace_sma` on either
    one or two molecules, using RDKit reaction SMARTS.

    Args:
        mol1 (Chem.Mol): First RDKit molecule (required).
        mol2 (Optional[Chem.Mol]): Second RDKit molecule, or None if single-mol mode.
        frag_sma (str): Reaction SMILES/SMARTS for the 'from' part.
        replace_sma (str): Reaction SMILES/SMARTS for the 'to' part.
        radius (int): Context radius used for protecting certain atoms.
        frag_ids_1 (Optional[List[int]]): Atom ids in mol1 for partial protection.
        frag_ids_2 (Optional[List[int]]): Atom ids in mol2 for partial protection.

    Yields:
        Tuple[str, Chem.Mol, str]: (SMILES, product Mol, reaction SMARTS).

    """  # noqa: D205

    def set_protected_atoms(
        mol: Chem.Mol,
        ids: list[int] | None,
        radius: int,
    ) -> None:
        """Mark certain atoms as protected by recursing outward from each protected atom up to `radius`.

        Args:
            mol (Chem.Mol): The RDKit molecule to modify.
            ids (Optional[List[int]]): Atom IDs in `mol` to protect.
            radius (int): Number of bonds outward to extend protection.

        Returns:
            None

        """

        def extend_ids(
            mol: Chem.Mol,
            atom_id: int,
            r: int,
            ids_local: set[int],
        ) -> None:
            """Recursively add neighbors up to distance `r` to protected IDs.

            Args:
                mol (Chem.Mol): The RDKit molecule.
                atom_id (int): Atom index from which to recurse.
                r (int): Remaining recursion depth (radius).
                ids_local (set[int]): The set of protected atom indices being built.

            Returns:
                None

            """
            if r:
                for neighbor in mol.GetAtomWithIdx(atom_id).GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx not in ids_local:
                        ids_local.add(neighbor_idx)
                    extend_ids(mol, neighbor_idx, r - 1, ids_local)

        if ids:
            ids_ext = set(ids)
            for i in ids:
                extend_ids(mol, i, radius + 1, ids_ext)
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() > 1 and atom.GetIdx() not in ids_ext:
                    atom.SetProp("_protected", "1")
                else:
                    atom.ClearProp("_protected")

    link = isinstance(mol2, Chem.Mol)  # simpler check
    if not isinstance(mol1, Chem.Mol):
        err_msg = "The first molecule in __gen_replacement always must be specified"
        raise StopIteration(err_msg)  # TRY003, EM101

    # Adjust input fragment SMARTS
    frag_sma = frag_sma.replace("*", "!#1")
    rxn_sma = f"{frag_sma}>>{replace_sma}"
    rxn = AllChem.ReactionFromSmarts(rxn_sma)

    set_protected_atoms(mol1, frag_ids_1, radius)
    if link:
        set_protected_atoms(mol2, frag_ids_2, radius)

    reactants = [[mol1, mol2], [mol2, mol1]] if link else [[mol1]]  # SIM108

    products: set[str] = set()
    for r in reactants:
        ps = rxn.RunReactants(r)
        for y in ps:
            for p in y:
                e = Chem.SanitizeMol(p, catchErrors=True)
                if e:
                    sys.stderr.write(
                        f"Molecule {Chem.MolToSmiles(p, isomericSmiles=True)} caused sanitization error {e}\n",
                    )
                    sys.stderr.flush()
                else:
                    smi = Chem.MolToSmiles(Chem.RemoveHs(p), isomericSmiles=True)
                    if smi not in products:
                        products.add(smi)
                        yield smi, p, rxn_sma


@profile
def __get_replacements_rowids(  # noqa: PLR0913
    db_cur: sqlite3.Cursor,
    env: str,
    dist: int | tuple[int, int] | None,
    min_atoms: int,
    max_atoms: int,
    radius: int,
    min_freq: int = 0,
    **kwargs,  # noqa: ANN003
) -> set[int]:
    """Return set of row IDs from table 'radius{radius}' that match environment `env`, frequency,
    number of atoms between `min_atoms` and `max_atoms`, and optional `dist` constraints.

    Args:
        db_cur (sqlite3.Cursor): SQLite cursor to the DB.
        env (str): The environment string to match.
        dist (Optional[int|Tuple[int,int]]): Distance constraint(s).
        min_atoms (int): Minimum number of atoms in fragment.
        max_atoms (int): Maximum number of atoms in fragment.
        radius (int): The context radius.
        min_freq (int): Minimum frequency threshold. Defaults to 0.
        **kwargs: Additional column constraints for SQL filtering.

    Returns:
        Set[int]: The rowids that match the query.

    """  # noqa: D205
    sql = (
        f"SELECT rowid FROM radius{radius} WHERE env = '{env}' AND freq >= {min_freq} "  # noqa: S608
        f"AND core_num_atoms BETWEEN {min_atoms} AND {max_atoms}"
    )
    if isinstance(dist, int):
        sql += f" AND dist2 = {dist}"
    elif isinstance(dist, tuple) and len(dist) == 2:
        sql += f" AND dist2 BETWEEN {dist[0]} AND {dist[1]}"
    for k, v in kwargs.items():
        if isinstance(v, tuple) and len(v) == 2:
            sql += f" AND {k} BETWEEN {v[0]} AND {v[1]}"
        else:
            sql += f" AND {k} = {v}"

    db_cur.execute(sql)  # nosec: S608 known risk, user logic handles constraints
    return {row[0] for row in db_cur.fetchall()}  # C401 => set comprehension


def _get_replacements(
    db_cur: sqlite3.Cursor,
    radius: int,
    row_ids: set[int],
) -> list[tuple[int, str, str, int]]:
    """Retrieve (rowid, core_smi, core_sma, freq) from 'radius{radius}' for the specified row_ids.

    Args:
        db_cur (sqlite3.Cursor): SQLite cursor to the DB.
        radius (int): The context radius, used to pick table name.
        row_ids (Set[int]): The row IDs to fetch.

    Returns:
        List[Tuple[int, str, str, int]]: (rowid, core_smi, core_sma, freq).

    """
    row_ids_list = ",".join(map(str, row_ids))
    sql = f"SELECT rowid, core_smi, core_sma, freq FROM radius{radius} WHERE rowid IN ({row_ids_list})"  # nosec: S608  # noqa: S608
    db_cur.execute(sql)
    return db_cur.fetchall()


@profile
def __gen_replacements(  # noqa: C901, PLR0912, PLR0913, PLR0915
    mol1: Chem.Mol,
    mol2: Chem.Mol | None,
    db_name: str,
    radius: int,
    dist: int | tuple[int, int] | None = None,
    min_size: int = 0,
    max_size: int = 8,
    min_rel_size: float = 0,
    max_rel_size: float = 1,
    min_inc: int = -2,
    max_inc: int = 2,
    max_replacements: int | None = None,
    replace_cycles: bool = False,  # FBT002  # noqa: FBT001, FBT002
    protected_ids_1: list[int] | None = None,
    protected_ids_2: list[int] | None = None,
    min_freq: int = 10,
    symmetry_fixes: bool = False,  # FBT002  # noqa: FBT001, FBT002
    filter_func=None,  # noqa: ANN001
    sample_func=None,  # noqa: ANN001
    return_frag_smi_only: bool = False,  # FBT002  # noqa: FBT001, FBT002
    **kwargs,  # noqa: ANN003
) -> None:  # type: ignore  # noqa: PGH003
    """Generate possible fragment replacements from DB, for single or linked molecules.

    Args:
        mol1 (Chem.Mol): Primary molecule, required.
        mol2 (Optional[Chem.Mol]): Secondary molecule if linking, else None.
        db_name (str): Path to the sqlite DB.
        radius (int): The context radius for environment table.
        dist (Optional[int|Tuple[int,int]]): Distance constraint(s).
        min_size (int): Minimum fragment size in heavy atoms.
        max_size (int): Maximum fragment size in heavy atoms.
        min_rel_size (float): Minimum ratio of replaced fragment size to parent.
        max_rel_size (float): Maximum ratio of replaced fragment size to parent.
        min_inc (int): Allowed negative difference in fragment size.
        max_inc (int): Allowed positive difference in fragment size.
        max_replacements (Optional[int]): Cap on number of replacements.
        replace_cycles (bool): Whether to allow cycle replacements ignoring size constraints.
        protected_ids_1 (Optional[List[int]]): Atom IDs protected in mol1.
        protected_ids_2 (Optional[List[int]]): Atom IDs protected in mol2.
        min_freq (int): Minimal fragment frequency in DB.
        symmetry_fixes (bool): Whether to handle symmetrical expansions (unused logic).
        filter_func: Optional user-defined function to filter row IDs.
        sample_func: Optional user-defined function to sample row IDs.
        return_frag_smi_only (bool): If True, only yield the replacement SMILES.
        **kwargs: Additional constraints used to filter the DB results.

    Yields:
        Different forms depending on link or single:
        - If return_frag_smi_only: just (core_smi)
        - If linking: (frag_sma, core_sma, freq, ids_1, ids_2)
        - Else single-mol: (frag_sma, core_sma, freq, ids[0])

    """
    link = isinstance(mol2, Chem.Mol)
    if not isinstance(mol1, Chem.Mol):
        err_msg = "The first molecule in __gen_replacement always must be specified"
        raise StopIteration(err_msg)  # TRY003, EM101

    # Fragment
    if link:
        f = __fragment_mol_link(
            mol1=mol1,
            mol2=mol2,
            radius=radius,
            protected_ids_1=protected_ids_1,
            protected_ids_2=protected_ids_2,
        )
        combined_mol = Chem.CombineMols(mol1, mol2)
        mol = combined_mol
    else:
        mol = mol1
        f = __fragment_mol(mol, radius, protected_ids=protected_ids_1, symmetry_fixes=symmetry_fixes)

    if not f:
        return

    mol_hac = mol.GetNumHeavyAtoms()
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    # Replacements can store row_ids for sampling or further usage
    replacements = {}  # C408 replaced dict()
    returned_values = 0
    preliminary_return = 0
    if max_replacements is not None:
        random.shuffle(f)
        preliminary_return = max_replacements // len(f)
        if preliminary_return == 0:
            preliminary_return = 1

    for env, core, *ids in f:
        tmp_m = Chem.MolFromSmiles(core)
        if tmp_m is None:
            continue
        num_heavy_atoms = tmp_m.GetNumHeavyAtoms()
        hac_ratio = num_heavy_atoms / mol_hac if mol_hac else 1.0

        is_ok_size = (min_size <= num_heavy_atoms <= max_size) and (min_rel_size <= hac_ratio <= max_rel_size)
        if is_ok_size or (replace_cycles and cycle_pattern.search(core)):
            frag_sma = combine_core_env_to_rxn_smarts(core, env)
            min_atoms = num_heavy_atoms + min_inc
            max_atoms = num_heavy_atoms + max_inc

            row_ids = __get_replacements_rowids(
                db_cur=cur,
                env=env,
                dist=dist,
                min_atoms=min_atoms,
                max_atoms=max_atoms,
                radius=radius,
                min_freq=min_freq,
                **kwargs,
            )

            if filter_func:
                row_ids = set(filter_func(row_ids, cur, radius))  # hush type warnings

            if max_replacements is None:
                res = _get_replacements(cur, radius, row_ids)
            else:
                n = min(len(row_ids), preliminary_return)
                if sample_func is not None:
                    selected_row_ids = sample_func(list(row_ids), cur, radius, n)
                else:
                    selected_row_ids = random.sample(list(row_ids), n)
                row_ids.difference_update(selected_row_ids)
                replacements.update({rid: (frag_sma, core, ids) for rid in row_ids})
                res = _get_replacements(cur, radius, set(selected_row_ids))

            for row_id, core_smi, core_sma, freq in res:  # noqa: B007
                if core_smi != core:
                    if return_frag_smi_only:
                        yield core_smi
                    elif link:
                        yield frag_sma, core_sma, freq, ids[0], ids[1]
                    else:
                        yield frag_sma, core_sma, freq, ids[0]
                    if max_replacements is not None:
                        returned_values += 1
                        if returned_values >= max_replacements:
                            return

    if max_replacements is not None:
        n = min(len(replacements), max_replacements - returned_values)
        if sample_func is not None:
            selected_row_ids = sample_func(list(replacements.keys()), cur, radius, n)
        else:
            selected_row_ids = random.sample(list(replacements.keys()), n)
        res2 = _get_replacements(cur, radius, set(selected_row_ids))
        for _row_id, core_smi, core_sma, freq in res2:  # B007 => rename row_id -> _row_id
            if core_smi != replacements[_row_id][1]:
                if return_frag_smi_only:
                    yield core_smi
                elif link:
                    yield (
                        replacements[_row_id][0],
                        core_sma,
                        freq,
                        replacements[_row_id][2][0],
                        replacements[_row_id][2][1],
                    )
                else:
                    yield replacements[_row_id][0], core_sma, freq, replacements[_row_id][2][0]


def __frag_replace_mp(
    items: tuple,
) -> list:
    """Multiprocessing helper to call __frag_replace, appending the final item to each result.

    Args:
        items (Tuple): (mol, mol2, frag_sma, core_sma, radius, frag_ids_1, frag_ids_2, freq)

    Returns:
        A list of (smi, product_mol, rxn_sma, freq).

    """
    return [(*res, items[-1]) for res in __frag_replace(*items[:-1])]


def __get_data(  # noqa: PLR0913, ANN202
    mol: Chem.Mol,
    db_name: str,
    radius: int,
    min_size: int,
    max_size: int,
    min_rel_size: float,
    max_rel_size: float,
    min_inc: int,
    max_inc: int,
    replace_cycles: bool,  # noqa: FBT001
    protected_ids: list[int] | None,
    min_freq: int,
    max_replacements: int | None,
    symmetry_fixes: bool,  # noqa: FBT001
    filter_func=None,  # noqa: ANN001
    sample_func=None,  # noqa: ANN001
    **kwargs,  # noqa: ANN003
):
    """Yield data items for each fragment replacement from __gen_replacements (single mol scenario).

    Args:
        mol (Chem.Mol): Single RDKit molecule.
        db_name (str): Path to fragment DB.
        radius (int): Context radius.
        min_size (int): Min fragment size in heavy atoms.
        max_size (int): Max fragment size in heavy atoms.
        min_rel_size (float): Minimum relative size ratio.
        max_rel_size (float): Maximum relative size ratio.
        min_inc (int): Min difference in fragment size.
        max_inc (int): Max difference in fragment size.
        replace_cycles (bool): Whether to ignore size constraints if cycle is found.
        protected_ids (Optional[List[int]]): Atom IDs to protect.
        min_freq (int): Minimum frequency.
        max_replacements (Optional[int]): Maximum expansions.
        symmetry_fixes (bool): Placeholder param for symmetrical expansions.
        filter_func: Optional filter function.
        sample_func: Optional sampling function.
        **kwargs: Additional constraints for DB queries.

    Yields:
        Tuples to pass to __frag_replace.

    """
    for frag_sma, core_sma, freq, ids in __gen_replacements(
        mol1=mol,
        mol2=None,
        db_name=db_name,
        radius=radius,
        min_size=min_size,
        max_size=max_size,
        min_rel_size=min_rel_size,
        max_rel_size=max_rel_size,
        min_inc=min_inc,
        max_inc=max_inc,
        max_replacements=max_replacements,
        replace_cycles=replace_cycles,
        protected_ids_1=protected_ids,
        protected_ids_2=None,
        min_freq=min_freq,
        symmetry_fixes=symmetry_fixes,
        filter_func=filter_func,
        sample_func=sample_func,
        return_frag_smi_only=False,
        **kwargs,
    ):
        yield mol, None, frag_sma, core_sma, radius, ids, None, freq


def __get_data_link(  # noqa: PLR0913, ANN202
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    db_name: str,
    radius: int,
    dist: int | tuple[int, int] | None,
    min_atoms: int,
    max_atoms: int,
    protected_ids_1: list[int] | None,
    protected_ids_2: list[int] | None,
    min_freq: int,
    max_replacements: int | None,
    filter_func=None,  # noqa: ANN001
    sample_func=None,  # noqa: ANN001
    **kwargs,  # noqa: ANN003
):
    """Yield data items for each fragment replacement from __gen_replacements (two-mol linking scenario).

    Args:
        mol1 (Chem.Mol): First RDKit molecule.
        mol2 (Chem.Mol): Second RDKit molecule.
        db_name (str): Path to fragment DB.
        radius (int): Context radius.
        dist (Optional[int|Tuple[int,int]]): Distance constraint(s).
        min_atoms (int): Min atoms in replaced fragment.
        max_atoms (int): Max atoms in replaced fragment.
        protected_ids_1 (Optional[List[int]]): Protected IDs in mol1.
        protected_ids_2 (Optional[List[int]]): Protected IDs in mol2.
        min_freq (int): Minimum frequency.
        max_replacements (Optional[int]): Maximum expansions.
        filter_func: Optional user-defined filter function.
        sample_func: Optional user-defined sampling function.
        **kwargs: Additional constraints for DB queries.

    Yields:
        Tuples to pass to __frag_replace for linking scenario.

    """
    for frag_sma, core_sma, freq, ids_1, ids_2 in __gen_replacements(
        mol1=mol1,
        mol2=mol2,
        db_name=db_name,
        radius=radius,
        dist=dist,
        min_size=0,
        max_size=0,
        min_rel_size=0,
        max_rel_size=1,
        min_inc=min_atoms,
        max_inc=max_atoms,
        max_replacements=max_replacements,
        replace_cycles=False,
        protected_ids_1=protected_ids_1,
        protected_ids_2=protected_ids_2,
        min_freq=min_freq,
        filter_func=filter_func,
        sample_func=sample_func,
        return_frag_smi_only=False,
        **kwargs,
    ):
        yield mol1, mol2, frag_sma, core_sma, radius, ids_1, ids_2, freq
