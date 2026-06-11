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
    UCR_*.txt
  multivariate/
    MSL/
    SMAP/
    SMD/
    PSM/
    NIPS_TS_GECCO/
    NIPS_TS_Swan/
    NIPS_TS_Creditcard/
```

## Run

Univariate UCR example:

```powershell
python src\run_univariate_ucr.py `
  --seeds 42 `
  --train_steps 15000 `
  --methods "Vanilla Data FM,Data FDM-lite,Data HCFM"
```

The univariate script runs all `UCR_*.txt` datasets found under `data/UCR`.

Multivariate audit / summary:

```powershell
python src\run_multivariate.py
```

## Outputs

Univariate results:

```text
outputs/univariate/run_<timestamp>/
```

Multivariate summaries:

```text
outputs/multivariate/
```
