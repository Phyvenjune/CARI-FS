# CARI-FS

Code for the paper *Causal-aware Framework for Robust and Interpretable Feature Selection*.

## Stages

1. `stage1_contribution.py`: feature contribution quantification with adaptive perturbation and Jensen-Shannon divergence.
2. `stage2_structure_paths.py`: causality-informed structure-path screening with weighted dependency scores, PC-style orientation, and Bootstrap stability screening.
3. `stage3_joint_response.py`: joint-perturbation response supplement for high-contribution candidates excluded by stage 2.

## Evaluation protocol

- Outer 80/20 train-test split.
- Internal 80/20 subtrain-validation split within outer training.
- Feature selection and early stopping use subtrain/validation only.
- The outer test split is evaluated once after the selected-feature model is trained.
- Multiclass F1 is reported as weighted F1, namely one-versus-rest F1 values weighted by class support.

## Run

```bash
python run_experiments.py --data-dir ../dataset --dataset all --output cari_fs_results.json
```

The default target column is the last CSV column. Use `--target` when needed.
