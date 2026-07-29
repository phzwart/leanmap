# Digits OOD detection (matched protocol)

Split: train=800, calib=200, test=400; α=0.05; seed=0.

| model | score | FPR | TPR gauss | TPR shuffle | AUC_g | digit singleton |
|---|---|---:|---:|---:|---:|---:|
| digits | cover | 0.080 | 1.000 | 1.000 | 0.999 | 0.945 |
| digits | affinity_entropy | 0.040 | 0.995 | 0.943 | 0.999 | 0.917 |
| digits | lda | 0.033 | 1.000 | 1.000 | 1.000 | 0.973 |
| digits_ood_basin_dual | cover | 0.068 | 0.905 | 1.000 | 0.981 | 0.767 |
| digits_ood_basin_dual | affinity_entropy | 0.052 | 0.993 | 0.943 | 0.997 | 0.932 |
| digits_ood_basin_dual | lda | 0.043 | 0.998 | 1.000 | 0.999 | 0.948 |
