import re
from collections import defaultdict
from itertools import product

from rdkit import Chem
from rdkit.Chem import rdMMPA

from crem.utils.mol_context import get_canon_context_core, patt_remove_map

cycle_pattern = re.compile(r"[a-zA-Z\]][1-9]+")
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
patt_remove_brackets = re.compile(r"\(\)")


def __extend_output_by_equivalent_atoms(mol, output):
    atom_ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False, includeIsotopes=False))
    tmp = defaultdict(list)
    for i, rank in enumerate(atom_ranks):
        tmp[rank].append(i)
    atom_eq = dict()
    for ids in tmp.values():
        if len(ids) > 1:
            for i in ids:
                atom_eq[i] = [j for j in ids if j != i]

    extended_output = []
    for item in output:
        if all(i in atom_eq for i in item[2]):
            smi = patt_remove_map.sub("", item[1])
            smi = patt_remove_brackets.sub("", smi)
            ids_list = [set(i) for i in mol.GetSubstructMatches(Chem.MolFromSmarts(smi))]
            for ids_matched in ids_list:
                for ids_eq in product(*(atom_eq[i] for i in item[2])):
                    if ids_matched == set(ids_eq):
                        extended_output.append((item[0], item[1], tuple(sorted(ids_eq))))
    return extended_output


def __fragment_mol(
    mol,
    radius=3,
    return_ids=True,
    keep_stereo=False,
    protected_ids=None,
    symmetry_fixes=False,
):
    def get_atom_prop(molecule, prop="Index"):
        res = []
        for a in molecule.GetAtoms():
            if a.GetAtomicNum():
                res.append(a.GetIntProp(prop))
        return tuple(sorted(res))

    if protected_ids:
        return_ids = True

    output = set()

    if return_ids:
        for atom in mol.GetAtoms():
            atom.SetIntProp("Index", atom.GetIdx())

    frags = rdMMPA.FragmentMol(mol, pattern="[!#1]!@!=!#[!#1]", maxCuts=4, resultsAsMols=True, maxCutBonds=30)
    frags += rdMMPA.FragmentMol(mol, pattern="[!#1]!@!=!#[!#1]", maxCuts=3, resultsAsMols=True, maxCutBonds=30)
    frags += rdMMPA.FragmentMol(mol, pattern="[#1]!@!=!#[!#1]", maxCuts=1, resultsAsMols=True, maxCutBonds=100)

    for i, (core, chains) in enumerate(frags):
        if core is None:
            components = list(Chem.GetMolFrags(chains, asMols=True))
            ids_0 = get_atom_prop(components[0]) if return_ids else tuple()
            ids_1 = get_atom_prop(components[1]) if return_ids else tuple()
            if Chem.MolToSmiles(components[0]) != "[H][*:1]":
                env, frag = get_canon_context_core(components[0], components[1], radius, keep_stereo)
                output.add((env, frag, ids_1))
            if Chem.MolToSmiles(components[1]) != "[H][*:1]":
                env, frag = get_canon_context_core(components[1], components[0], radius, keep_stereo)
                output.add((env, frag, ids_0))
        else:
            env, frag = get_canon_context_core(chains, core, radius, keep_stereo)
            output.add((env, frag, get_atom_prop(core) if return_ids else tuple()))

    if symmetry_fixes:
        extended_output = __extend_output_by_equivalent_atoms(mol, output)
        if extended_output:
            output.update(extended_output)

    if protected_ids:
        protected_ids = set(protected_ids)
        output = [item for item in output if protected_ids.isdisjoint(item[2])]

    return list(output)


def __fragment_mol_link(
    mol1,
    mol2,
    radius=3,
    keep_stereo=False,
    protected_ids_1=None,
    protected_ids_2=None,
    return_ids=True,
):
    def filter_frags(frags, protected_ids):
        output = []
        protected_ids = set(protected_ids)
        for _, chains in frags:
            for atom in chains.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    for d in atom.GetNeighbors():
                        if d.GetAtomicNum() != 1 and d.GetIdx() not in protected_ids:
                            output.append((None, chains))
        return output

    def prep_frags(frags, keep_stereo=False):
        ls = []
        for _, chains in frags:
            ids = []
            for atom in chains.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    for d in atom.GetNeighbors():
                        if d.GetAtomicNum() == 1:
                            ids = [d.GetIntProp("Index")]
                if ids:
                    break
            a, b = Chem.MolToSmiles(chains, isomericSmiles=keep_stereo).split(".")
            if a == "[H][*:1]":
                ls.append([b, ids])
            else:
                ls.append([a, ids])
        return ls

    if protected_ids_1 or protected_ids_2:
        return_ids = True

    if return_ids:
        for atom in mol1.GetAtoms():
            atom.SetIntProp("Index", atom.GetIdx())
        for atom in mol2.GetAtoms():
            atom.SetIntProp("Index", atom.GetIdx())

    frags_1 = rdMMPA.FragmentMol(mol1, pattern="[#1]!@!=!#[!#1]", maxCuts=1, resultsAsMols=True, maxCutBonds=100)
    frags_2 = rdMMPA.FragmentMol(mol2, pattern="[#1]!@!=!#[!#1]", maxCuts=1, resultsAsMols=True, maxCutBonds=100)

    if protected_ids_1:
        frags_1 = filter_frags(frags_1, protected_ids_1)

    if protected_ids_2:
        frags_2 = filter_frags(frags_2, protected_ids_2)

    frags_1 = prep_frags(frags_1, keep_stereo)
    frags_2 = prep_frags(frags_2, keep_stereo)

    for i in range(len(frags_1)):
        frags_1[i][0] = frags_1[i][0].replace("*:1", "*:2")

    q = []
    for (fr1, ids1), (fr2, ids2) in product(frags_1, frags_2):
        q.append(["%s.%s" % (fr1, fr2), ids1, ids2])

    fake_core = "[*:1]C[*:2]"
    output = []

    for chains, ids_1, ids_2 in q:
        env, frag = get_canon_context_core(chains, fake_core, radius=radius, keep_stereo=keep_stereo)
        output.append((env, "[H][*:1].[H][*:2]", ids_1, ids_2))

    return output
