#!/usr/bin/env python3
# ==============================================================================
# author          : Pavel Polishchuk
# date            : 26-06-2019
# version         :
# python_version  : 3.8+
# copyright       : Pavel Polishchuk 2019
# license         :
# ==============================================================================

import argparse
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from time import time

import joblib
import numpy as np
import pandas as pd
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from joblib import delayed
from line_profiler import profile
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rich.logging import RichHandler

# -- RICH IMPORTS --
from rich.traceback import install as install_rich_traceback

# IMPORTANT:
# Make sure this import is from your actual local crem code, e.g.:
# from crem.crem import mutate_mol2
# or relative import if your structure is different.
from crem.core.operations import mutate_mol2


def setup_rich_logger(log_level: int = logging.INFO) -> logging.Logger:
    """Create and configure a custom logger that uses Rich for pretty, colorful logging.

    Args:
        log_level: The desired log level, e.g. logging.INFO, logging.DEBUG, etc.

    Returns:
        A configured logging.Logger instance using RichHandler.

    """
    # Rich can show nicer, colorized tracebacks
    install_rich_traceback(show_locals=False)

    # Create a new logger with a RichHandler
    # format="%(message)s" ensures RichHandler does its default styling
    logging.basicConfig(
        level=log_level,
        format="%(message)s",  # Let Rich handle the formatting style
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Optionally hush RDKit warnings to a certain level:
    # logging.getLogger("rdkit").setLevel(logging.WARNING)

    return logger


def make_mating_pool(
    population_mol: Sequence[Mol],
    population_scores: Sequence[float],
    offspring_size: int,
) -> NDArray[Mol]:
    """Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights.

    Args:
        population_mol: Sequence of RDKit Mol
        population_scores: Sequence of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return

    Returns:
        NDArray of RDKit Mol (probably not unique)

    """
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(
        population_mol,
        p=population_probs,
        size=offspring_size,
        replace=True,
    )
    return mating_pool


def score_mol(mol: Mol, score_fn) -> float:
    """Score a single molecule using the provided scoring function."""
    return score_fn(Chem.MolToSmiles(mol))


class CREM_Generator(GoalDirectedGenerator):
    def __init__(
        self,
        smi_file: str | Path,
        selection_size: int,
        db_fname: str | Path,
        radius: int,
        replacements: int,
        max_size: int,
        min_size: int,
        max_inc: int,
        min_inc: int,
        generations: int,
        ncpu: int,
        random_start: bool,
        output_dir: str | Path,
    ):
        """Initialize the CREM_Generator with all needed parameters."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.pool = joblib.Parallel(n_jobs=ncpu)
        self.smiles = self.load_smiles_from_file(smi_file)
        self.N = selection_size
        self.db_fname = db_fname
        self.radius = radius
        self.min_size = min_size
        self.max_size = max_size
        self.min_inc = min_inc
        self.max_inc = max_inc
        self.replacements = replacements
        self.replacements_baseline = replacements
        self.generations = generations
        self.random_start = random_start
        self.patience1 = 3
        self.patience2 = 10
        self.patience3 = 33
        self.task = 0
        self.output_dir = Path(output_dir)

    def load_smiles_from_file(self, smi_file: str | Path) -> list[str]:
        """Load SMILES strings from a file."""
        self.logger.debug(f"Loading SMILES from file: {smi_file}")
        with open(smi_file) as f:
            lines = [line.strip() for line in f]
        self.logger.info(f"Loaded {len(lines)} SMILES from file.")
        return lines

    def top_k(self, smiles: Sequence[str], scoring_function: ScoringFunction, k: int) -> list[str]:
        """Score each SMILES, then sort them and return the top k."""
        self.logger.debug(f"Scoring {len(smiles)} SMILES to find top {k}.")
        joblist = (delayed(scoring_function.score)(s) for s in smiles)
        scores = self.pool(joblist)
        scored_smiles = sorted(zip(scores, smiles, strict=False), key=lambda x: x[0], reverse=True)
        top_smiles = [smile for score, smile in scored_smiles][:k]
        self.logger.debug(f"Top {len(top_smiles)} SMILES selected.")
        return top_smiles

    @profile
    def generate(self, smiles: Sequence[str]) -> list[str]:
        """Given a list of SMILES, generate new molecules using mutate_mol2 from the CREM library."""
        self.logger.debug(f"Generating new molecules from {len(smiles)} SMILES...")
        mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in smiles]
        res = self.pool(
            delayed(mutate_mol2)(
                mol,
                db_name=self.db_fname,
                radius=self.radius,
                min_size=self.min_size,
                max_size=self.max_size,
                min_rel_size=0,
                max_rel_size=1,
                min_inc=self.min_inc,
                max_inc=self.max_inc,
                max_replacements=self.replacements,
                replace_cycles=False,
                protected_ids=None,
                min_freq=0,
                return_rxn=False,
                return_rxn_freq=False,
                ncores=1,
            )
            for mol in mols
        )
        # Flatten nested lists, remove duplicates
        new_smiles = list(set(m for sublist in res for m in sublist))
        self.logger.debug(f"Generated {len(new_smiles)} new unique SMILES.")
        return new_smiles

    def set_params(self, score: float) -> None:
        """Adjust generation parameters based on the current best score."""
        self.logger.debug(f"Adjusting parameters based on best score of {score:.3f}")
        self.replacements = self.replacements_baseline
        if score > 0.8:
            self.min_inc, self.max_inc = -4, 4
        elif score > 0.7:
            self.min_inc, self.max_inc = -5, 5
        elif score > 0.6:
            self.min_inc, self.max_inc = -6, 6
        elif score > 0.5:
            self.min_inc, self.max_inc = -7, 7
        elif score > 0.4:
            self.min_inc, self.max_inc = -8, 8
        elif score > 0.3:
            self.min_inc, self.max_inc = -9, 9
        else:
            self.min_inc, self.max_inc = -10, 10

    def get_scores(self, scoring_function: ScoringFunction, smiles: Sequence[str]) -> list[float]:
        """Convert SMILES to RDKit Mols and score them with the given scoring function."""
        self.logger.debug(f"Scoring {len(smiles)} SMILES.")
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        return self.pool(delayed(score_mol)(m, scoring_function.score) for m in mols)

    def generate_optimized_molecules(
        self,
        scoring_function: ScoringFunction,
        number_molecules: int,
        starting_population: list[str] | None = None,
    ) -> list[str]:
        """Main evolutionary loop:
        1. Initialize or take a given starting population.
        2. Generate new molecules, score them, keep the top scorers.
        3. Update parameters, repeat.
        """
        self.task += 1
        self.logger.info(f"Starting generation process for task #{self.task}")
        self.logger.info(f"Requested number of molecules: {number_molecules}")

        if number_molecules > self.N:
            self.N = number_molecules
            self.logger.warning(
                "Benchmark requested more molecules than expected. " f"New population size is {number_molecules}",
            )

        # Select initial population
        if starting_population is None:
            self.logger.info("Selecting initial population...")
            if self.random_start:
                initial_smiles = np.random.choice(self.smiles, self.N)
                self.logger.debug(f"Randomly selected {self.N} SMILES as initial population.")
            else:
                initial_smiles = self.top_k(self.smiles, scoring_function, self.N)
                self.logger.debug(f"Selected top {self.N} SMILES from the input database.")
            population = pd.DataFrame({"smi": initial_smiles})
        else:
            population = pd.DataFrame({"smi": starting_population})
            self.logger.info(f"Using provided starting population of size {len(population)}.")

        # Calculate initial scores
        population["score"] = self.get_scores(scoring_function, population["smi"])

        # Evolution loop setup
        t0 = time_start = time()
        patience1 = patience2 = patience3 = 0
        best = population.copy()
        ref_score = np.mean(best["score"].iloc[:number_molecules])
        self.set_params(best["score"].max())
        used_smiles = set(population["smi"])

        self.logger.info(f"Initial best average score: {ref_score:.3f}")

        for generation in range(self.generations):
            if ref_score == 1.0:
                self.logger.info(f"Perfect score reached at generation {generation}. Stopping early.")
                break

            self.logger.debug(f"Generating new population (gen {generation})...")
            new_smiles = self.generate(population["smi"])
            population = pd.DataFrame({"smi": new_smiles})
            population["score"] = self.get_scores(scoring_function, population["smi"])

            # Combine old best + new population, remove duplicates, keep top N
            combined = pd.concat([best, population], ignore_index=True)
            combined = combined.drop_duplicates(subset="smi")
            combined = combined.sort_values(by="score", ascending=False)
            best = combined.head(self.N)

            # Evaluate the new best average score
            cur_score = np.mean(best["score"].iloc[:number_molecules])
            self.logger.debug(f"Current score for generation {generation}: {cur_score:.3f}")

            if cur_score > ref_score:
                # We found improvement
                ref_score = cur_score
                population = population.head(self.N)
                self.set_params(population["score"].max())
                used_smiles.update(population["smi"])
                patience1 = patience2 = patience3 = 0
                self.logger.info(
                    f"Improvement found at generation {generation}. New best avg: {ref_score:.3f}",
                )
            else:
                # No improvement, increment patience counters
                patience1 += 1
                patience2 += 1
                patience3 += 1

                self.logger.debug(
                    f"No improvement at generation {generation}. "
                    f"Patience counters: p1={patience1}, p2={patience2}, p3={patience3}",
                )

                if patience3 >= self.patience3:
                    self.logger.info(f"Maximum patience (p3) reached at generation {generation}.")
                    if starting_population is None and self.random_start:
                        self.logger.warning("Resetting population from random SMILES due to p3 threshold.")
                        patience1 = patience2 = patience3 = 0
                        initial_smiles = np.random.choice(self.smiles, self.N)
                        population = pd.DataFrame({"smi": initial_smiles})
                        population["score"] = self.get_scores(scoring_function, population["smi"])
                        population = population.sort_values("score", ascending=False)
                        self.set_params(population["score"].max())
                        used_smiles = set(population["smi"])
                    else:
                        # Stop the evolutionary loop
                        self.logger.info(f"Stopping evolutionary loop at generation {generation}.")
                        break
                else:
                    # Keep only top N
                    population = population.head(self.N)
                    used_smiles.update(population["smi"])

                    if patience2 >= self.patience2:
                        # Medium-level patience threshold
                        self.logger.info("p2 threshold reached; relaxing parameters further.")
                        patience1 = patience2 = 0
                        self.min_inc -= 10
                        self.max_inc += 10
                        self.replacements += 500
                    elif patience1 >= self.patience1:
                        # Small patience threshold
                        self.logger.info("p1 threshold reached; adjusting parameters slightly.")
                        patience1 = 0
                        self.min_inc -= 1
                        self.max_inc += 1
                        self.replacements += 100

            # Print generation statistics
            gen_time = time() - t0
            t0 = time()
            self.logger.info(
                f"Gen {generation:3d} | best avg: {np.mean(best['score'].iloc[:number_molecules]):.3f} "
                f"| max: {population['score'].max():.3f} | avg: {population['score'].mean():.3f} "
                f"| min: {population['score'].min():.3f} | std: {population['score'].std():.3f} "
                f"| sum: {population['score'].sum():.3f} "
                f"| min_inc: {self.min_inc} | max_inc: {self.max_inc} | repl: {self.replacements} "
                f"| p1: {patience1} | p2: {patience2} | p3: {patience3} | {gen_time:.2f} sec",
            )

            # 5-hour time limit
            if t0 - time_start > 18000:
                self.logger.warning(
                    f"Time limit (5 hours) reached. Stopping early at generation {generation}.",
                )
                break

        # Save results to a .smi file
        output_path = Path(self.output_dir) / f"{self.task}.smi"
        best.round({"score": 3}).to_csv(
            output_path,
            sep="\t",
            header=False,
            index=False,
        )
        self.logger.info(f"Saved best molecules for task #{self.task} to {output_path}")

        # Return the top (number_molecules) SMILES
        return best["smi"].iloc[:number_molecules].tolist()


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_file", type=str, required=True)
    parser.add_argument("--db_fname", type=str, required=True)
    parser.add_argument("--selection_size", type=int, default=10)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--replacements", type=int, default=1000)
    parser.add_argument("--min_size", type=int, default=0)
    parser.add_argument("--max_size", type=int, default=10)
    parser.add_argument("--min_inc", type=int, default=-7)
    parser.add_argument("--max_inc", type=int, default=7)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--ncpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--suite", default="v2")
    parser.add_argument("--log_level", type=str, default="INFO", help="Set log level (DEBUG, INFO, WARNING, ERROR).")

    args = parser.parse_args()

    # Convert string log_level to a logging constant, e.g. "INFO" -> logging.INFO
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_rich_logger(log_level=numeric_level)

    np.random.seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Determine output directory
    output_dir = Path(args.output_dir or os.path.dirname(os.path.realpath(__file__)))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the command-line args for reference
    param_file = output_dir / "goal_directed_params.json"
    with open(param_file, "w") as jf:
        json.dump(vars(args), jf, sort_keys=True, indent=4)
    logger.info(f"Saved parameters to {param_file}")

    # Create optimizer
    optimiser = CREM_Generator(
        smi_file=args.smiles_file,
        selection_size=args.selection_size,
        db_fname=args.db_fname,
        radius=args.radius,
        min_size=args.min_size,
        max_size=args.max_size,
        min_inc=args.min_inc,
        max_inc=args.max_inc,
        replacements=args.replacements,
        generations=args.generations,
        ncpu=args.ncpu,
        random_start=True,
        output_dir=output_dir,
    )

    # JSON output for the benchmark
    json_file_path = output_dir / "goal_directed_results.json"
    logger.info(f"Running GuacaMol assessment with suite={args.suite}")

    # Evaluate the generator using GuacaMol's goal-directed benchmark suite
    assess_goal_directed_generation(
        optimiser,
        json_output_file=json_file_path,
        benchmark_version=args.suite,
    )
    logger.info(f"Finished all benchmarks. Results saved to {json_file_path}")


if __name__ == "__main__":
    entry_point()
