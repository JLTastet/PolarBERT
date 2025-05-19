_Updated: 2025-04-14_
## Fine-tuning and transfer

Pretrain model on dataset X, fine-tune on Y for task T, evaluate on Z, without time offset
where:
	X $\in$ Kaggle (130M), Kaggle (350k), Prometheus (350k)
	Y, Z $\in$ Kaggle
	T $\in$ direction or energy reconstruction
and optimising the fine-tuning hyperparameters in each case.
Compare to baselines trained directly on the downstream task.

The experiments below are performed on the `v0.3` version of the code.
### Pretraining

Pretrained checkpoints (X):
- Kaggle (130M) $\to$ `checkpoints/results/kaggle-130M-tuned-v2/Flash Transformer/last.ckpt`
- Kaggle (350k) $\to$ `checkpoints/results/kaggle-350k-tuned/kaggle-tuned-350k_events_250409-160721/last.ckpt`
- Prometheus (350k) $\to$ `checkpoints/results/prometheus-tuned-v3/prometheus-tuned-v3_250410-035606/last.ckpt`
### Supervised baselines

Prometheus-100k:
- Sweep: https://wandb.ai/polargeese/PolarBERT-from_scratch-sweeps/sweeps/z80hy8k1?nw=nwuserjltastet
- Performance is random (loss ~1.57) when training on 100k events only.

Kaggle-100k:
- Sweep: https://wandb.ai/polargeese/PolarBERT-from_scratch-sweeps/sweeps/yrswi9p9?nw=nwuserjltastet
- Performance is nearly random (loss ~1.53) when training on 100k events only. The model probably picks up the up/down asymmetry, hence why the loss isn’t 1.57.

Kaggle-130M:
- [Sweep](https://wandb.ai/polargeese/PolarBERT-from_scratch-sweeps/sweeps/fulxgond?nw=nwuserjltastet), [Run](https://wandb.ai/polargeese/PolarBERT-direction_from_scratch/runs/v4l8zeyo?nw=nwuserjltastet)
- Checkpoint: `training_from_scratch/kaggle-130Mevt-tuned_250508-213104/last.ckpt`

### Fine-tuning

Ideally we would re-tune the fine-tuning hyperparameters for each base model and task, but this is inefficient. Instead let’s check that the optimal hyperparameters don’t deviate too much between base models and task, compared to a reference run (directional fine-tuning of model pretrained on 130M Kaggle events, using 100k fine-tuning events).

Reference run (Kaggle-130M $\to$ Kaggle-100k)
- [Sweep](https://wandb.ai/polargeese/PolarBERT-finetuning-sweeps/sweeps/82n117sh?nw=nwuserjltastet) [Run](https://wandb.ai/polargeese/PolarBERT-finetuning-results/runs/irzyjcsw?nw=nwuserjltastet)
- Checkpoint: `checkpoints/results/directional-kaggle_130M_on_kaggle_100k-tuned/directional-kaggle_130M_on_kaggle_100k-tuned_250410-182609/last.ckpt`

Varying the checkpoint (Kaggle-350k $\to$ Kaggle-100k)
- [Sweep](https://wandb.ai/polargeese/PolarBERT-finetuning-sweeps/sweeps/a6urgz3n?nw=nwuserjltastet) [Run](https://wandb.ai/polargeese/PolarBERT-finetuning-results/runs/978d2pzi?nw=nwuserjltastet)
- Checkpoint: `checkpoints/results/directional-kaggle_350k_on_kaggle_100k-tuned_250411-130107/last.ckpt`

Varying the fine-tuning dataset (Kaggle-130M $\to$ Prometheus-100k)
- [Sweep](https://wandb.ai/polargeese/PolarBERT-finetuning-sweeps/sweeps/00vwd6xz/workspace?nw=nwuserjltastet) [Run](https://wandb.ai/polargeese/PolarBERT-finetuning-results/runs/ttao2udg?nw=nwuserjltastet)
- Checkpoint: `checkpoints/results/directional-kaggle_130M_on_prometheus_100k-tuned_250411-125345/last.ckpt`

The optimal fine-tuning hyperparameters seem roughly consistent, apart from the number of epochs which is a bit larger when starting from the Kaggle-350k checkpoint.
To take care of this, hyperparameters that aren’t well constrained should be allowed to vary around their preferred value.
Let’s implement this by using a common configuration and setting a gaussian prior on parameters that are not fully constrained yet.
After the sweeps, all hyperparameters except `max_epochs`, `max_lr` and `weight_decay` seem to have common optimal values. The medians among the top few runs are used for the remaining three hyperparameters.

Remaining fine-tuning runs:
- Kaggle-350k $\to$ Prometheus-100k
  [Sweep](https://wandb.ai/polargeese/PolarBERT-finetuning-sweeps/sweeps/znk84wjv?nw=nwuserjltastet) [Run](https://wandb.ai/polargeese/PolarBERT-finetuning-results/runs/o3f8sz8k?nw=nwuserjltastet)
  Checkpoint: `checkpoints/results/directional-kaggle_350k_on_prometheus_100k-tuned_250414-165826/last.ckpt`
  Note: some minor overfitting of hyperparameters was observed for this sweep, such that the validation loss of the retrained model is slightly worse than for the best run from the sweep.
- Prometheus-350k $\to$ Prometheus-100k
  [Sweep](https://wandb.ai/polargeese/PolarBERT-finetuning-sweeps/sweeps/xfre8ose?nw=nwuserjltastet) [Run](https://wandb.ai/polargeese/PolarBERT-finetuning-results/runs/gpb8mm2x?nw=nwuserjltastet)
  Checkpoint: `checkpoints/results/directional-prometheus_350k_on_prometheus_100k-tuned_250414-165826/last.ckpt`
- Prometheus-350k $\to$ Kaggle-100k
  [Sweep](https://wandb.ai/polargeese/PolarBERT-finetuning-sweeps/sweeps/pumnozfx?nw=nwuserjltastet) [Run](https://wandb.ai/polargeese/PolarBERT-finetuning-results/runs/pybkl0wn?nw=nwuserjltastet)
  Checkpoint: `checkpoints/results/directional-prometheus_350k_on_kaggle_100k-tuned_250414-165826/last.ckpt`
#### Summary tables

Validation loss after fine-tuning on the angular reconstruction task.

| $\downarrow$ Pretrained / Fine-tuned $\rightarrow$ | Kaggle (100k) | Prometheus (100k) |
| -------------------------------------------------- | ------------- | ----------------- |
| Kaggle (130M)                                      | 1.08          | 0.96              |
| Kaggle (350k)                                      | 1.26          | 1.19              |
| Prometheus (350k)                                  | 1.24          | 1.18              |
- We can observed various degrees of transfer learning in all cases. However, we have so far only evaluated the models on the task they were trained for. Another type of transfer learning would be to evaluate them on the other dataset (e.g. if fine-tuned on Kaggle, evaluate them on Prometheus).
- We can also note that only models trained on Kaggle (130M) perform better than naive linear regression (~1.2 loss if I remember correctly from the Kaggle competition).
- Overall, when controlling for the number of events, models trained and/or evaluated on Prometheus have a lower angular loss.

| Supervised baseline | Angular loss on same dataset |
| ------------------- | ---------------------------- |
| Kaggle (130M)       | 1.03*                        |
| Kaggle (100k)       | 1.54                         |
| Prometheus (100k)   | 1.55                         |
| [Kaggle winner](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/discussion/402976) | 0.96 |
| [Line fit](https://www.kaggle.com/code/solverworld/icecube-picks-points-with-least-squares) | 1.18 |

*(\* = still training)*

Transferring the fine-tuned model to the other dataset (on which it wasn’t fine-tuned), we obtain the following losses:

| $\downarrow$ Pretrained / Fine-tuned (transferred to) $\rightarrow$ | Kaggle (100k)<br>(transferred to Prometheus) | Prometheus (100k)<br>(transferred to Kaggle) |
| ------------------------------------------------------------------- | -------------------------------------------- | -------------------------------------------- |
| Kaggle (130M)                                                       | 1.44                                         | 1.46                                         |
| Kaggle (350k)                                                       | 1.48                                         | 1.51                                         |
| Prometheus (350k)                                                   | 1.47                                         | 1.52                                         |

- Very minor transfer seems to be happening (better than a random guess), but the performance regression is large compared to the dataset on which the model was fine-tuned.

| Supervised baseline | Transferred to | Angular loss on other dataset |
| ------------------- | -------------- | ----------------------------- |
| Kaggle (130M)       | Prometheus     | 1.45*                         |
| Kaggle (100k)       | Prometheus     | N/A                           |
| Prometheus (100k)   | Kaggle         | N/A                           |

*(\* = still training)*

- Event the supervised baseline trained on 130M Kaggle events does not generalise to the Prometheus dataset.
