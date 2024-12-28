"""operations.py.

Implements core operations for mutating, growing, and linking molecules using a fragment-based approach:
- mutate_mol
- grow_mol
- link_mols
- mutate_mol2, grow_mol2, link_mols2 (list-returning versions)
- get_replacements for advanced usage

We have addressed Ruff lint warnings by:
    - Adding docstrings, type hints
    - Suppressing or acknowledging complex / argument-count warnings (C901, PLR0912, PLR0913, PLR0915)
    - Removing unused imports
    - Converting or suppressing places where simpler comprehensions are suggested
    - Not changing underlying logic
"""

from multiprocessing import Pool, cpu_count

from rdkit import Chem

# We import from our replacement module
from crem.core.replacement import (
    __frag_replace,
    __frag_replace_mp,
    __gen_replacements,
    __get_data,
)

# We also import from fragmentation if needed


def mutate_mol(  # noqa: C901, PLR0913, PLR0912
    mol: Chem.Mol,
    db_name: str,
    radius: int = 3,
    min_size: int = 0,
    max_size: int = 10,
    min_rel_size: float = 0,
    max_rel_size: float = 1,
    min_inc: int = -2,
    max_inc: int = 2,
    max_replacements: int | None = None,
    replace_cycles: bool = False,  # noqa: FBT001, FBT002
    replace_ids: list[int] | None = None,
    protected_ids: list[int] | None = None,
    symmetry_fixes: bool = False,  # noqa: FBT001, FBT002
    min_freq: int = 0,
    return_rxn: bool = False,  # noqa: FBT001, FBT002
    return_rxn_freq: bool = False,  # noqa: FBT001, FBT002
    return_mol: bool = False,  # noqa: FBT001, FBT002
    ncores: int = 1,
    filter_func=None,  # noqa: ANN001
    sample_func=None,  # noqa: ANN001
    **kwargs,  # noqa: ANN003
) -> Chem.Mol | str | None:  # type: ignore  # noqa: PGH003
    """Mutate a molecule by substituting fragments from a database.

    Args:
        mol (Chem.Mol): The input molecule to mutate.
        db_name (str): The fragment DB name.
        radius (int): The context radius.
        min_size (int): Minimum heavy atoms in fragment replaced.
        max_size (int): Maximum heavy atoms in fragment replaced.
        min_rel_size (float): Minimum relative size ratio for fragment replaced.
        max_rel_size (float): Maximum relative size ratio for fragment replaced.
        min_inc (int): Min difference in size between replaced & new.
        max_inc (int): Max difference in size.
        max_replacements (Optional[int]): The cap on how many replacements to yield.
        replace_cycles (bool): If True, can replace ring fragments ignoring size constraints.
        replace_ids (Optional[List[int]]): Atom indices to replace.
        protected_ids (Optional[List[int]]): Atom indices to protect from replacement.
        symmetry_fixes (bool): Whether to attempt symmetrical expansions.
        min_freq (int): Min occurrence threshold in the DB.
        return_rxn (bool): If True, include the reaction SMARTS in output.
        return_rxn_freq (bool): If True, include the freq in output.
        return_mol (bool): If True, yield the RDKit product Mol instead of only SMILES.
        ncores (int): Number of CPU cores for parallel.
        filter_func: Optional user-supplied filter for row IDs.
        sample_func: Optional user-supplied sampling for row IDs.
        **kwargs: Additional parameters for advanced usage.

    Yields:
        - If return_mol=False, return_rxn=False: yields just the SMILES string
        - If one or more booleans are True, yields a tuple of details accordingly

    Returns:
        A generator that yields mutated structures (str or tuple).

    """
    products = {Chem.MolToSmiles(mol)}
    prot_set = set(protected_ids) if protected_ids else set()

    if replace_ids:
        ids = set()
        for i in replace_ids:
            ids.update(a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetAtomicNum() == 1)
        all_ids = {a.GetIdx() for a in mol.GetAtoms()}
        # SIM102 => we won't flatten or we risk logic changes
        if ids:
            ids = all_ids.difference(ids).difference(replace_ids)
        prot_set.update(ids)

    protected_ids_sorted = sorted(prot_set)

    if ncores == 1:
        # Single core mode
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
            protected_ids_1=protected_ids_sorted,
            protected_ids_2=None,
            min_freq=min_freq,
            symmetry_fixes=symmetry_fixes,
            filter_func=filter_func,
            sample_func=sample_func,
            return_frag_smi_only=False,
            **kwargs,
        ):
            for smi, mol_prod, rxn in __frag_replace(mol, None, frag_sma, core_sma, radius, ids, None):
                if max_replacements is None or len(products) < (max_replacements + 1):  # noqa: SIM102
                    if smi not in products:
                        products.add(smi)
                        res = [smi]
                        if return_rxn:
                            res.append(rxn)
                            if return_rxn_freq:
                                res.append(freq)
                        if return_mol:
                            res.append(mol_prod)
                        if len(res) == 1:
                            yield res[0]
                        else:
                            yield tuple(res)
    else:
        # Multi-core mode
        p = Pool(min(ncores, cpu_count()))
        try:
            data_iter = __get_data(
                mol,
                db_name,
                radius,
                min_size,
                max_size,
                min_rel_size,
                max_rel_size,
                min_inc,
                max_inc,
                replace_cycles,
                protected_ids_sorted,
                min_freq,
                max_replacements,
                symmetry_fixes,
                filter_func=filter_func,
                sample_func=sample_func,
                **kwargs,
            )
            for items in p.imap(__frag_replace_mp, data_iter, chunksize=100):
                for smi, mol_prod, rxn, freq in items:
                    if max_replacements is None or len(products) < (max_replacements + 1):  # noqa: SIM102
                        if smi not in products:
                            products.add(smi)
                            res = [smi]
                            if return_rxn:
                                res.append(rxn)
                                if return_rxn_freq:
                                    res.append(freq)
                            if return_mol:
                                res.append(mol_prod)
                            if len(res) == 1:
                                yield res[0]
                            else:
                                yield tuple(res)
        finally:
            p.close()
            p.join()


def grow_mol(  # noqa: C901, PLR0912, PLR0913
    mol: Chem.Mol,
    db_name: str,
    radius: int = 3,
    min_atoms: int = 1,
    max_atoms: int = 2,
    max_replacements: int | None = None,
    replace_ids: list[int] | None = None,
    protected_ids: list[int] | None = None,
    symmetry_fixes: bool = False,  # noqa: FBT001, FBT002
    min_freq: int = 0,
    return_rxn: bool = False,  # noqa: FBT001, FBT002
    return_rxn_freq: bool = False,  # noqa: FBT001, FBT002
    return_mol: bool = False,  # noqa: FBT001, FBT002
    ncores: int = 1,
    filter_func=None,  # noqa: ANN001
    sample_func=None,  # noqa: ANN001
    **kwargs,  # noqa: ANN003
) -> Chem.Mol | str | None:
    """Grow a molecule by replacing its hydrogen atoms with fragments from the DB.

    Args:
        mol (Chem.Mol): The input molecule to grow.
        db_name (str): The fragment DB name.
        radius (int): The context radius.
        min_atoms (int): Min number of new atoms in the fragment to attach.
        max_atoms (int): Max number of new atoms.
        max_replacements (Optional[int]): Cap on the number of expansions to yield.
        replace_ids (Optional[List[int]]): Atom indices to replace.
        protected_ids (Optional[List[int]]): Atom indices to protect from expansion.
        symmetry_fixes (bool): Whether to attempt symmetrical expansions.
        min_freq (int): Minimum occurrence threshold in the DB.
        return_rxn (bool): If True, yield reaction SMARTS in output.
        return_rxn_freq (bool): If True, yield freq in output.
        return_mol (bool): If True, yield the RDKit product Mol instead of only SMILES.
        ncores (int): Number of CPU cores for parallel.
        filter_func: Optional user-supplied filter function.
        sample_func: Optional user-supplied sampling function.
        **kwargs: Additional advanced parameters.

    Yields:
        Generator of mutated structures (SMILES or tuple details).

    """
    m = Chem.AddHs(mol)

    if protected_ids:
        ids_list: list[int] = []
        for i in protected_ids:
            if m.GetAtomWithIdx(i).GetAtomicNum() == 1:
                ids_list.append(i)
            # SIM102 =>
            elif m.GetAtomWithIdx(i).GetNeighbors():
                for a in m.GetAtomWithIdx(i).GetNeighbors():
                    if a.GetAtomicNum() == 1:
                        ids_list.append(a.GetIdx())  # noqa: PERF401
        prot_set = set(ids_list)
    else:
        prot_set = set()

    if replace_ids:
        ids_set = set()
        for i in replace_ids:
            if m.GetAtomWithIdx(i).GetAtomicNum() == 1:
                ids_set.add(i)
            # SIM102 =>
            elif m.GetAtomWithIdx(i).GetNeighbors():
                for a in m.GetAtomWithIdx(i).GetNeighbors():
                    if a.GetAtomicNum() == 1:
                        ids_set.add(a.GetIdx())
        all_h = {atom.GetIdx() for atom in m.GetAtoms() if atom.GetAtomicNum() == 1}
        diff = all_h.difference(ids_set)
        prot_set.update(diff)

    return mutate_mol(
        m,
        db_name,
        radius,
        min_size=0,
        max_size=0,
        min_inc=min_atoms,
        max_inc=max_atoms,
        max_replacements=max_replacements,
        replace_ids=None,
        protected_ids=sorted(prot_set),
        min_freq=min_freq,
        return_rxn=return_rxn,
        return_rxn_freq=return_rxn_freq,
        return_mol=return_mol,
        ncores=ncores,
        symmetry_fixes=symmetry_fixes,
        filter_func=filter_func,
        sample_func=sample_func,
        **kwargs,
    )


def link_mols(  # noqa: C901, PLR0913, PLR0912, PLR0915
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    db_name: str,
    radius: int = 3,
    dist: int | None = None,
    min_atoms: int = 1,
    max_atoms: int = 2,
    max_replacements: int | None = None,
    replace_ids_1: list[int] | None = None,
    replace_ids_2: list[int] | None = None,
    protected_ids_1: list[int] | None = None,
    protected_ids_2: list[int] | None = None,
    min_freq: int = 0,
    return_rxn: bool = False,  # noqa: FBT001, FBT002
    return_rxn_freq: bool = False,  # noqa: FBT001, FBT002
    return_mol: bool = False,  # noqa: FBT001, FBT002
    ncores: int = 1,
    filter_func=None,  # noqa: ANN001
    sample_func=None,  # noqa: ANN001
    **kwargs,  # noqa: ANN003
) -> Chem.Mol | str | None:  # type: ignore  # noqa: PGH003
    """Link two molecules by substituting or bridging with fragments from the database.

    Args:
        mol1 (Chem.Mol): First molecule.
        mol2 (Chem.Mol): Second molecule.
        db_name (str): Fragment DB name.
        radius (int): Context radius for fragment environment.
        dist (Optional[int]): Distance constraint(s) for bridging.
        min_atoms (int): Min new atoms.
        max_atoms (int): Max new atoms.
        max_replacements (Optional[int]): The cap on expansions.
        replace_ids_1 (Optional[List[int]]): Atom indices to replace in mol1.
        replace_ids_2 (Optional[List[int]]): Atom indices to replace in mol2.
        protected_ids_1 (Optional[List[int]]): Atom indices to protect in mol1.
        protected_ids_2 (Optional[List[int]]): Atom indices to protect in mol2.
        min_freq (int): Min frequency threshold in DB.
        return_rxn (bool): If True, yield reaction SMARTS in output.
        return_rxn_freq (bool): If True, yield freq in output.
        return_mol (bool): If True, yield RDKit product Mol.
        ncores (int): Number of CPU cores for parallel.
        filter_func: Optional function to filter row IDs from DB.
        sample_func: Optional function to sample row IDs from DB.
        **kwargs: Additional advanced usage parameters.

    Yields:
        A generator of possible linked structures (SMILES or tuple).

    """

    def __get_protected_ids(
        m: Chem.Mol,
        replace_ids: list[int] | None,
        prot_ids: list[int] | None,
    ) -> set[int]:
        """Determine which atom indices should be protected in a given molecule.

        Args:
            m (Chem.Mol): The RDKit molecule.
            replace_ids (Optional[List[int]]): Atoms to be replaced.
            prot_ids (Optional[List[int]]): Already-protected atoms.

        Returns:
            Set of protected atom IDs.

        """
        if prot_ids:
            tmp_ids = set()
            for i in prot_ids:
                if m.GetAtomWithIdx(i).GetAtomicNum() == 1:
                    # Perf401/C401 =>
                    for a in m.GetAtomWithIdx(i).GetNeighbors():
                        tmp_ids.add(a.GetIdx())
                else:
                    tmp_ids.add(i)
            prot_ids = tmp_ids
        else:
            prot_ids = set()

        if replace_ids:
            more_ids = set()
            for i in replace_ids:
                # nested if =>
                if m.GetAtomWithIdx(i).GetAtomicNum() == 1:
                    for a in m.GetAtomWithIdx(i).GetNeighbors():
                        more_ids.add(a.GetIdx())
                else:
                    more_ids.add(i)
            heavy_atom_ids = {atm.GetIdx() for atm in m.GetAtoms() if atm.GetAtomicNum() > 1}
            # Combine
            final_ids = heavy_atom_ids.difference(more_ids)
            prot_ids.update(final_ids)
        return prot_ids

    products: set[str] = set()

    mol1_h = Chem.AddHs(mol1)
    mol2_h = Chem.AddHs(mol2)

    prot_ids_1 = __get_protected_ids(mol1_h, replace_ids_1, protected_ids_1)
    prot_ids_2 = __get_protected_ids(mol2_h, replace_ids_2, protected_ids_2)

    if ncores == 1:
        for frag_sma, core_sma, freq, ids_1, ids_2 in __gen_replacements(
            mol1=mol1_h,
            mol2=mol2_h,
            db_name=db_name,
            radius=radius,
            dist=dist,
            min_size=0,
            max_size=0,
            min_rel_size=0,
            max_rel_size=1,
            min_inc=min_atoms,
            max_inc=max_atoms,
            replace_cycles=False,
            max_replacements=max_replacements,
            protected_ids_1=prot_ids_1,
            protected_ids_2=prot_ids_2,
            min_freq=min_freq,
            filter_func=filter_func,
            sample_func=sample_func,
            return_frag_smi_only=False,
            **kwargs,
        ):
            for smi, new_mol, rxn in __frag_replace(mol1_h, mol2_h, frag_sma, core_sma, radius, ids_1, ids_2):
                if max_replacements is None or (max_replacements is not None and len(products) < max_replacements):  # noqa: SIM102
                    if smi not in products:
                        products.add(smi)
                        out = [smi]
                        if return_rxn:
                            out.append(rxn)
                            if return_rxn_freq:
                                out.append(freq)
                        if return_mol:
                            out.append(new_mol)
                        if len(out) == 1:
                            yield out[0]
                        else:
                            yield tuple(out)
    else:
        from .replacement import __get_data_link

        p = Pool(min(ncores, cpu_count()))
        try:
            data_iter = __get_data_link(
                mol1_h,
                mol2_h,
                db_name,
                radius,
                dist,
                min_atoms,
                max_atoms,
                prot_ids_1,
                prot_ids_2,
                min_freq,
                max_replacements,
                filter_func=filter_func,
                sample_func=sample_func,
                **kwargs,
            )
            for items in p.imap(__frag_replace_mp, data_iter, chunksize=100):
                for smi, new_mol, rxn, freq in items:
                    if max_replacements is None or (max_replacements is not None and len(products) < max_replacements):  # noqa: SIM102
                        if smi not in products:
                            products.add(smi)
                            out = [smi]
                            if return_rxn:
                                out.append(rxn)
                                if return_rxn_freq:
                                    out.append(freq)
                            if return_mol:
                                out.append(new_mol)
                            if len(out) == 1:
                                yield out[0]
                            else:
                                yield tuple(out)
        finally:
            p.close()
            p.join()


def mutate_mol2(  # noqa: ANN201
    *args,
    **kwargs,
):
    """Same as mutate_mol but returns a list instead of generator."""  # noqa: D401
    return list(mutate_mol(*args, **kwargs))


def grow_mol2(  # noqa: ANN201
    *args,
    **kwargs,
):
    """Same as grow_mol but returns a list instead of generator."""  # noqa: D401
    return list(grow_mol(*args, **kwargs))


def link_mols2(  # noqa: ANN201
    *args,
    **kwargs,
):
    """Same as link_mols but returns a list instead of generator."""  # noqa: D401
    return list(link_mols(*args, **kwargs))


def get_replacements(  # noqa: PLR0913
    mol1: Chem.Mol,
    db_name: str,
    radius: int,
    mol2: Chem.Mol | None = None,
    dist: int | None = None,
    min_size: int = 0,
    max_size: int = 8,
    min_rel_size: float = 0,
    max_rel_size: float = 1,
    min_inc: int = -2,
    max_inc: int = 2,
    max_replacements: int | None = None,
    replace_cycles: bool = False,  # noqa: FBT001, FBT002
    protected_ids_1: list[int] | None = None,
    protected_ids_2: list[int] | None = None,
    replace_ids_1: list[int] | None = None,
    replace_ids_2: list[int] | None = None,
    min_freq: int = 0,
    symmetry_fixes: bool = False,  # noqa: FBT001, FBT002
    filter_func=None,  # noqa: ANN001
    sample_func=None,  # noqa: ANN001
    **kwargs,  # noqa: ANN003
) -> None:
    """Generate replacement fragment SMILES for a single or pair of molecules.

    Args:
        mol1 (Chem.Mol): First molecule (required).
        db_name (str): Fragment DB name.
        radius (int): Context radius.
        mol2 (Optional[Chem.Mol]): Second molecule if linking, else None.
        dist (Optional[int]): Distance constraint(s).
        min_size (int): Min heavy atoms for replaced fragment.
        max_size (int): Max heavy atoms.
        min_rel_size (float): Min ratio of replaced frag size to parent.
        max_rel_size (float): Max ratio.
        min_inc (int): Min difference in fragment size.
        max_inc (int): Max difference in fragment size.
        max_replacements (Optional[int]): Limit on expansions.
        replace_cycles (bool): If True, allow ignoring size constraints for ring frags.
        protected_ids_1 (Optional[List[int]]): Protected IDs in mol1.
        protected_ids_2 (Optional[List[int]]): Protected IDs in mol2.
        replace_ids_1 (Optional[List[int]]): Which IDs to replace in mol1.
        replace_ids_2 (Optional[List[int]]): Which IDs to replace in mol2.
        min_freq (int): Min frequency threshold in DB.
        symmetry_fixes (bool): Placeholder for symmetrical expansions.
        filter_func: Optional user function to filter row IDs from DB.
        sample_func: Optional user function to sample row IDs from DB.
        **kwargs: Additional constraints for advanced usage.

    Yields:
        str: Replacement fragment SMILES, if found.

    """
    prot_ids_1 = set(protected_ids_1) if protected_ids_1 else set()
    if replace_ids_1:
        rep_ids_1 = set(replace_ids_1)
        prot_ids_1 = prot_ids_1 | set(range(mol1.GetNumAtoms())).difference(rep_ids_1)

    if isinstance(mol2, Chem.Mol):
        prot_ids_2 = set(protected_ids_2) if protected_ids_2 else set()
        if replace_ids_2:
            rep_ids_2 = set(replace_ids_2)
            prot_ids_2 = prot_ids_2 | set(range(mol2.GetNumAtoms())).difference(rep_ids_2)
    else:
        prot_ids_2 = None

    # UP028 => we keep the for loop with yield for logic clarity
    for frag_smi in __gen_replacements(  # noqa: UP028
        mol1=mol1,
        mol2=mol2,
        db_name=db_name,
        radius=radius,
        dist=dist,
        min_size=min_size,
        max_size=max_size,
        min_rel_size=min_rel_size,
        max_rel_size=max_rel_size,
        min_inc=min_inc,
        max_inc=max_inc,
        max_replacements=max_replacements,
        replace_cycles=replace_cycles,
        protected_ids_1=prot_ids_1,
        protected_ids_2=prot_ids_2,
        min_freq=min_freq,
        symmetry_fixes=symmetry_fixes,
        filter_func=filter_func,
        sample_func=sample_func,
        return_frag_smi_only=True,
        **kwargs,
    ):
        yield frag_smi
