_Updated: 2025-05-09_
## Fine-tuning and transfer with time offset

Pretrain model on dataset X, fine-tune on Y for task T, evaluate on Z, with a random time offset of 1.5 added during fine-tuning and evaluation.
where:
	X $\in$ Kaggle (130M), Kaggle (350k), Prometheus (350k)
	Y, Z $\in$ Kaggle
	T $\in$ direction reconstruction
and optimising the fine-tuning hyperparameters in each case. `random_time_offset: 1.5` is added to YAML config file.
Compare to baselines trained directly on the downstream task (also with the time offset).

### Pretraining

Pretrained checkpoints (X) must be re-tuned now that we are including a time offset.

Kaggle (130M)
- [Sweep](https://wandb.ai/polargeese/PolarBERT-sweeps/sweeps/qvcv183g?nw=nwuserjltastet), [Run](https://wandb.ai/polargeese/PolarBERT-results/runs/5doc1vn8?nw=nwuserjltastet)
- Checkpoint: `checkpoints/results/kaggle-130M-time_offset-tuned_250506-012336/last.ckpt`
- Very unstable training, required multiple sweeps of increasing granularity, over all hyperparameters, to get reliable training.

Kaggle (350k)
- [Sweep](https://wandb.ai/polargeese/PolarBERT-sweeps/sweeps/2l72klhg?nw=nwuserjltastet), [Run](https://wandb.ai/polargeese/PolarBERT-results/runs/hxlad6rt?nw=nwuserjltastet)
- Checkpoint: `checkpoints/results/kaggle-350k-time_offset-tuned/kaggle-tuned-350k_events-randtime-1.5_250422-183803/last.ckpt`

Prometheus (350k)
- [Sweep](https://wandb.ai/polargeese/PolarBERT-sweeps/sweeps/anh2umje?nw=nwuserjltastet), [Run](https://wandb.ai/polargeese/PolarBERT-results/runs/xe455tq4?nw=nwuserjltastet)
- Checkpoint: `checkpoints/results/prometheus-time_offset-tuned_250426-215311/last.ckpt`

### Baselines (with time offset)

Training from scratch on the smaller Kaggle-100k and Prometheus-100k datasets already didn’t work without a time offset, so it doesn’t make sense to try with one.

Kaggle-130M:
- [Sweep](https://wandb.ai/polargeese/PolarBERT-from_scratch-sweeps/sweeps/tgt1yodv?nw=nwuserjltastet), [Run](https://wandb.ai/polargeese/PolarBERT-direction_from_scratch/runs/0qlggntp?nw=nwuserjltastet) (exploded, need to further tune HPs and retrain)
- Best runs from the sweep achieve an angular loss of ~1.04.
- I can probably brute-force more stable hyperparameters by running a finer sweep.

### Fine-tuning (with time offset)

Kaggle-350k $\to$ Prometheus-100k
- [Sweep](https://wandb.ai/polargeese/PolarBERT-finetuning-sweeps/sweeps/lbrmbevx?nw=nwuserjltastet)
- Almost doesn’t train (gets stuck in the local minimum ~1.53).
- Still doesn’t train if we only apply the time offset during pretraining, but not during fine-tuning.

Kaggle-130M $\to$ Prometheus-100k
- [Sweep](https://wandb.ai/polargeese/PolarBERT-finetuning-sweeps/sweeps/6markrg6/workspace?nw=nwuserjltastet), [Run](https://wandb.ai/polargeese/PolarBERT-finetuning-results/runs/q8sjeh66?nw=nwuserjltastet)
- Best validation loss 1.38

Kaggle-130M $\to$ Kaggle-100k
- [Sweep](https://wandb.ai/polargeese/PolarBERT-finetuning-sweeps/sweeps/pxgokmtl/workspace?nw=nwuserjltastet)
- Best validation loss 1.32

Kaggle-130M $\to$ Kaggle-1M
- [Sweep](https://wandb.ai/polargeese/PolarBERT-finetuning-sweeps/sweeps/4kwt1ldw/workspace?nw=nwuserjltastet)
- Best validation loss **so far** 1.21 (ongoing sweep)

### Summary tables (with time offset)

For the results below, the random offset is disabled for evaluation.

| $\downarrow$ Pretrained / Fine-tuned $\rightarrow$ | Kaggle (100k) | Kaggle (1M) | Kaggle (10M) | Prometheus (100k) |
| -------------------------------------------------- | ------------- | ----------- | ------------ | ----------------- |
| Kaggle (130M)                                      | 1.32          | 1.13        | 1.07         | 1.38              |
| Kaggle (350k)                                      |               |             |              | 1.53              |
| Prometheus (350k)                                  |               |             |              |                   |

| Supervised baseline | Angular loss |
| ------------------- | ------------ |
| Kaggle (130M)       | 1.04 $^\dagger$ |

*($\dagger$ = best run hard to reproduce, could be solved by more comprehensive sweep)

### Transfer evaluation (with time offset)

| $\downarrow$ Pretrained / Fine-tuned (transferred to) $\rightarrow$ | Kaggle (100k)<br>(transferred to Prometheus) | Prometheus (100k)<br>(transferred to Kaggle) |
| ------------------------------------------------------------------- | -------------------------------------------- | -------------------------------------------- |
| Kaggle (130M)                                                       |                                              | 1.50                                         |
| Kaggle (350k)                                                       |                                              | N/A                                          |
| Prometheus (350k)                                                   |                                              |                                              |

| Supervised baseline | Angular loss on Prometheus (100k) |
| ------------------- | --------------------------------- |
| Kaggle (130M)       | (need to train a new checkpoint)  |

## Procedure for Pretraining (Kaggle-350k with Time Offset Example)

*(The text below is LLM-generated)*

This section outlines the steps followed to pretrain a model on the Kaggle-350k dataset subset with a random time offset, including hyperparameter tuning.

**1. Preparation:**

*   **Identify Base Configs:** Start with the existing tuned config for the baseline experiment (e.g., `configs/polarbert-kaggle-350k-tuned.yaml`) and the corresponding sweep config (e.g., `configs/polarbert-kaggle-350k-fine-sweep.yaml`).
*   **Define Experiment Name:** Choose a consistent naming scheme (e.g., using `time_offset` with underscores).

**2. Create Sweep Files:**

*   **Create Untuned Config:**
    *   Copy the base *tuned* config to a new file named using the experiment scheme (e.g., `configs/polarbert-kaggle-350k-time_offset-untuned.yaml`).
    *   Add/modify experiment-specific parameters (e.g., `training.random_time_offset: 1.5`).
    *   Update `model.model_name` (e.g., append `-untuned-time_offset`).
    *   Change `training.project` to the sweeps project (e.g., `PolarBERT-sweeps`).
    *   Set `training.checkpoint.dirpath` to a temporary sweep directory (e.g., `checkpoints/pretrain-kaggle-350k-time_offset-sweep`).
    *   Disable checkpoint saving for sweeps: set `training.checkpoint.save_top_k: 0`, `save_last: false`, `save_final: false`.
*   **Create Sweep Config:**
    *   Copy the base *sweep* config to a new file (e.g., `configs/polarbert-kaggle-350k-time_offset-sweep.yaml`).
    *   Update the `name` field for the sweep.
    *   In the `command` section, update the `--config` path to point to the newly created *untuned* config file (relative path from `src/polarbert`, e.g., `../../configs/polarbert-kaggle-350k-time_offset-untuned.yaml`).
    *   Remove command-line arguments that are now handled by the config (like `--random_time_offset`). Ensure the parameter exists in the referenced untuned config.
    *   Adjust hyperparameter ranges (`parameters`) if needed.

**3. Run Hyperparameter Sweep:**

*   **Generate Sweep ID:** Run `wandb sweep configs/polarbert-kaggle-350k-time_offset-sweep.yaml` from the workspace root. Note the full sweep path (e.g., `polargeese/PolarBERT-sweeps/xxxxxxxx`).
*   **Create Sweep Slurm Script:**
    *   Copy the template (`scripts/slurm.sh`) or an existing sweep script to a new file (e.g., `scripts/slurm-pretrain-kaggle-350k-time_offset-sweep.sh`).
    *   Set `#SBATCH --job-name` to something descriptive (e.g., `pre-offK-S`).
    *   Set `#SBATCH --output` to the correct log path, using the consistent naming scheme (e.g., `../../logs/pretrain-kaggle-350k-time_offset-sweep-%j.log`).
    *   Ensure `#SBATCH --nice=999` for sweep priority.
    *   Remove any `cd` commands if submitting from `src/polarbert`.
    *   Update the `wandb agent` command with the correct full sweep path and use `--count 1` for pretraining sweeps.
*   **Submit Sweep Agents:** From the `src/polarbert` directory, run `sbatch ../../scripts/slurm-pretrain-kaggle-350k-time_offset-sweep.sh` multiple times to launch parallel agents.
*   **Monitor Sweep:** Check progress on the W&B dashboard.

**4. Analyze Sweep and Generate Tuned Config:**

*   **Fetch Results:** Once the sweep is complete, run the fetch script from the workspace root: `python scripts/fetch_sweep_results.py`. Ensure `SWEEP_PATH` and `OUTPUT_FILENAME` inside the script are updated for the current sweep.
*   **Analyze and Generate Config:** Run the analysis script from the workspace root: 
    ```bash
    python scripts/analyze_sweep_results.py \
        --csv_file tables/sweep_results_pretrain_kaggle_350k_time_offset.csv \
        --top_k 5 \
        --input_config configs/polarbert-kaggle-350k-time_offset-untuned.yaml \
        --output_config configs/polarbert-kaggle-350k-time_offset-tuned.yaml
    ```
    (Adjust `--top_k` or use `--quantile` as needed. Ensure paths use the correct naming scheme).
    This script calculates median hyperparameters for the best runs, prints them, and automatically generates the tuned config file, ensuring correct data types (e.g., int for epochs/batch size) and updating project name, model name, and checkpoint settings.

**5. Run Final Tuned Training:**

*   **Create Tuned Slurm Script:**
    *   Copy the sweep Slurm script or template to a new file (e.g., `scripts/slurm-pretrain-kaggle-350k-time_offset-tuned.sh`).
    *   Set `#SBATCH --job-name` (e.g., `pre-offK-T`).
    *   Set `#SBATCH --output` log path correctly (e.g., `../../logs/pretrain-kaggle-350k-time_offset-tuned-%j.log`).
    *   Ensure `#SBATCH --nice=0` for normal priority.
    *   Replace the `wandb agent` command with the training command:
        ```bash
        srun python pretraining.py \
            --config ../../configs/polarbert-kaggle-350k-time_offset-tuned.yaml \
            --model_type flash \
            --dataset_type kaggle 
        ```
        (Ensure the `--config` path is correct relative to `src/polarbert`. Note that `--random_time_offset` is no longer needed as it's read from the config).
*   **Submit Tuned Job:** From the `src/polarbert` directory, run `sbatch ../../scripts/slurm-pretrain-kaggle-350k-time_offset-tuned.sh`.
*   **Monitor:** Check job status (`squeue`) and log file (`logs/...`). The final checkpoint will be saved in the `dirpath` specified in the tuned config (e.g., `checkpoints/results/pretrain-kaggle-350k-time_offset-tuned/`).

### Potential Pitfalls

*   **Naming Consistency:** Ensure consistent use of hyphens vs. underscores (e.g., `time_offset` vs `time-offset`) across related filenames (configs, scripts) and internal references (log paths, config paths). Underscores are currently used for the time offset experiment.
*   **Relative Paths:** Python scripts (`pretraining.py`, `finetuning.py`) seem to interpret config paths relative to the execution directory (`src/polarbert`), requiring paths like `../../configs/...`. SBATCH log paths are relative to the script location, but work as `../../logs/...` when submitting from `src/polarbert`.
*   **YAML Data Types:** Hyperparameters like `max_epochs` and `logical_batch_size` must be integers in the final config. The analysis script (`analyze_sweep_results.py`) needs to explicitly cast these after calculating the median.
*   **Sweep Config Dependencies:** Always update both the sweep config (`*_sweep.yaml`) and the base *untuned* config it points to, especially for non-swept parameters, paths, and names.
*   **Checkpoint Saving:** Disable saving in the *untuned* config used for sweeps; enable it in the final *tuned* config.
*   **W&B Project:** Use `...-sweeps` project for sweeps and `...-results` for final runs.
*   **Slurm `cd`:** Remove `cd` commands from Slurm scripts if submitting from `src/polarbert`.
*   **Slurm `nice`:** Use `999` for sweeps, `0` for single runs.
*   **Sweep Agent Count:** Use `--count 1` for pretraining sweep agents.
*   **Sweep ID:** Remember to manually insert the correct sweep ID into the Slurm sweep script after running `wandb sweep`.
*   **Script Paths:** When running utility scripts (`fetch`, `analyze`) from the workspace root, use paths relative to the root (e.g., `scripts/...`, `configs/...`, `tables/...`). 