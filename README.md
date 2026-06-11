# HCFM Code Submission

## Setup

```bash
pip install -r requirements.txt
```

## Layout

```text
src/        benchmark code and HCFM implementation
scripts/    PowerShell launch helpers
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
.\scripts\run_ucr_full_example.ps1
```

Equivalent direct command:

```powershell
python src\run_univariate_ucr.py `
  --datasets UCR_1,UCR_16,UCR_21,UCR_45,UCR_120 `
  --seeds 42 `
  --train_steps 15000 `
  --methods "Vanilla Data FM,Data FDM-lite,Data HCFM" `
  --score_profile core `
  --full_divergence
```

Multivariate audit / summary:

```powershell
.\scripts\run_multivariate_audit.ps1
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

