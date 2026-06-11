# HCFM Code Submission

## Setup

```bash
pip install -r requirements.txt
```

## Layout

```text
src/        benchmark code and HCFM implementation
data/       place datasets here
outputs/    generated results
```

Expected data layout:

```text
data/
  UCR/
    UCR_*_train.npy
    UCR_*_test.npy
    UCR_*_test_label.npy
  multivariate/
    MSL/
    SMAP/
    SMD/
    PSM/
    NIPS_TS_GECCO/
    NIPS_TS_Swan/
    NIPS_TS_Creditcard/
```

Datasets are available here:
https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR?usp=sharing

## Run

Univariate UCR example:

```powershell
python src\run_univariate_ucr.py `
  --seeds 42 `
  --train_steps 15000 `
```

The univariate script runs all UCR datasets found under `data/UCR`.

Multivariate benchmark:

```powershell
python src\run_multivariate.py `
  --seeds 42 `
  --train_steps 15000
```

The multivariate script runs all known datasets found under `data/multivariate`
by default. Use `--dataset_id MSL` to run one dataset.

## Outputs

Univariate results:

```text
outputs/univariate/run_<timestamp>/
```

Multivariate results:

```text
outputs/multivariate/
```
