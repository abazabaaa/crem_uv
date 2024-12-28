"""test_api.py.

A minimal script showing how to exercise the CReM high-level API:
 - mutate
 - grow
 - link
"""

from crem.api import grow, mutate


def main():
    # Path to your SQLite fragment DB (adjust to your environment)
    db_path = "/Users/thomasgraham/prj/crem/output_uv/fragments.db"

    # 1. Mutate example
    input_smiles = "c1ccccc1C"  # Toluene
    mutated = mutate(
        mol=input_smiles,
        db_path=db_path,
        radius=3,
        max_replacements=5,
        ring_only=True,  # e.g., let's let the ring be replaced with ring fragments
        ncores=1,
    )
    print("Mutated structures (up to 5):")
    for i, smi in enumerate(mutated, start=1):
        print(f"  {i:2d}. {smi}")

    # 2. Grow example: add new fragments at hydrogen positions
    input_smiles_grow = "CC1=C(C(C)=CC=C1)CNC2=NCCN(C2)CC3=CC=CC=N3"  # Ethanol
    grown = grow(
        mol=input_smiles_grow,
        db_path=db_path,
        radius=3,
        min_atoms=2,
        max_atoms=3,
        max_replacements=5,
        ncores=1,
    )
    print("\nGrown structures (up to 5):")
    for i, smi in enumerate(grown, start=1):
        print(f"  {i:2d}. {smi}")


if __name__ == "__main__":
    main()
