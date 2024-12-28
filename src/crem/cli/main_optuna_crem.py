import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import optuna
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.utils.helpers import setup_default_logger

from crem.guacamol_crem_test import CREM_Generator


def crem_optuna_objective(trial: optuna.trial.Trial) -> float:
    """This objective function runs a short iterative search (like from the CReM paper)
    with user-tunable parameters. 
    We'll measure how well the generated molecules achieve some internal "score."
    
    Return a single scalar to be *maximized*. 
    """
    # 1. Let trial pick hyperparameters
    #    (We rely heavily on CReM's approach: radius, replacements, inc, fragment occurrence, etc.)
    radius = trial.suggest_int("radius", 1, 5)
    replacements = trial.suggest_int("replacements", 100, 2000, step=100)
    # optional frequency cutoff
    min_freq = trial.suggest_int("min_freq", 0, 1000, step=50)
    # maybe we vary the generation approach
    generations = trial.suggest_int("generations", 50, 200, step=50)

    # 2. We'll create a temporary output dir for each trial
    temp_dir = Path(tempfile.mkdtemp(prefix="crem_optuna_trial_"))

    # 3. Create a CREM_Generator with these parameters
    #    (This is the code from your guacamol_crem_test.py, with fields overridden)
    generator = CREM_Generator(
        smi_file = "/Users/thomasgraham/prj/crem/example/CHEMBL231.smi",
        selection_size=10,
        db_fname="/Users/thomasgraham/prj/crem/db_output/fragments.db",   # The pre-built fragment DB from step #1
        radius=radius,
        replacements=replacements,
        min_size=0,
        max_size=10,
        min_inc=-7,
        max_inc=7,
        generations=generations,
        ncpu=10,
        random_start=True,
        output_dir=temp_dir,
    )
    # Optionally set the 'min_freq' in the generator, if the code supports it:
    generator.min_freq = min_freq

    # 4. We do a *short* assess_goal_directed_generation
    #    - E.g., we rely on GuacaMol to measure "overall_score" for a single benchmark,
    #      or you define a custom measure.
    setup_default_logger()
    out_json = temp_dir / "trial_results.json"

    # For demonstration, we run a single (or small subset) GuacaMol tasks for speed:
    try:
        assess_goal_directed_generation(
            generator,
            json_output_file=out_json,
            benchmark_version="v2",  # or "v3"
        )
    except Exception as e:
        print(f"[WARNING] Trial failed: {e}")
        # Return something to indicate a low score
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 0.0

    # 5. Parse the "trial_results.json" for a final 'score'
    try:
        with open(out_json) as f:
            data = json.load(f)
        # The GuacaMol JSON typically has a structure with a 'score' field,
        # or an array of tasks. We'll do a simple approach:
        #
        # The data might look like:
        #   {"benchmarks": [{"name": "...", "score": 0.82, ...}, {"name": "..."}],
        #    "metadata": {...}
        #   }
        # So we can sum or average the "score" fields:
        bench_scores = [b["score"] for b in data["benchmarks"]]
        score = float(np.mean(bench_scores))  # or sum, etc.
    except:
        score = 0.0

    # 6. Clean up
    shutil.rmtree(temp_dir, ignore_errors=True)

    return score  # to be *maximized* by Optuna

def run_optuna_for_crem():
    study = optuna.create_study(
        study_name="TuneCREM",
        direction="maximize",  # we want the best guacamol-based score
        # Optionally: storage="sqlite:///crem_optuna.db", load_if_exists=True
    )
    study.optimize(crem_optuna_objective, n_trials=20, n_jobs=1)

    print("Best Trial:")
    print(f"  Number: {study.best_trial.number}")
    print(f"  Value (score): {study.best_trial.value}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k} = {v}")

def final_crem_run_with_best(study):
    best_params = study.best_trial.params
    # Suppose we only have radius, replacements, min_freq, generations
    # We'll do a bigger run:

    final_generator = CREM_Generator(
        smi_file         = "example_input_smiles.smi",
        db_fname         = "path/to/your_crem_fragdb.pkl",
        radius           = best_params["radius"],
        replacements     = best_params["replacements"],
        min_size         = 0,
        max_size         = 10,
        min_inc          = -7,
        max_inc          = 7,
        generations      = best_params["generations"],
        ncpu             = 2,
        random_start     = True,
        output_dir       = Path("final_crem_output"),
    )
    final_generator.min_freq = best_params["min_freq"]

    out_json = Path("final_results.json")
    setup_default_logger()
    assess_goal_directed_generation(
        final_generator,
        json_output_file=out_json,
        benchmark_version="v2",
    )
    print("[INFO] Final run results in final_results.json")

def main():
    # Possibly generate or load your fragment DB first if needed:
    #   smiles_for_db = load_my_chembl_subset()  # or whichever method
    #   generate_crem_db(smiles_for_db, "path/to/your_crem_fragdb.pkl", radius=3)

    # 1. Run Optuna
    run_optuna_for_crem()

    # 2. Suppose we want to do a final run
    #    We can retrieve the study from memory or from the storage
    #    Then pass it to final_crem_run_with_best(study)
    #    e.g. final_crem_run_with_best(study)
    #
    # If run_optuna_for_crem returns a study object:
    #    study = run_optuna_for_crem()
    #    final_crem_run_with_best(study)


if __name__ == "__main__":
    main()
