$ErrorActionPreference = "Stop"

python src\run_univariate_ucr.py `
  --datasets UCR_1,UCR_16,UCR_21,UCR_45,UCR_120 `
  --seeds 42 `
  --train_steps 15000 `
  --methods "Vanilla Data FM,Data FDM-lite,Data HCFM" `
  --score_profile core `
  --full_divergence
