# RQ1 · Closed-pool results

Each cell reports the mean ± sample standard deviation across five seeds. **P** is response quality, **C** is normalized cost, and **R** is reward, computed as R = P − λC. Each split contains 557 training, 119 validation, and 121 test queries. All methods within a split share the cost scale and test queries. The splits reuse the same corpus, so the standard deviations describe variation across runs.

| Method | P (λ = 0.1) | C (λ = 0.1) | R (λ = 0.1) | P (λ = 0.5) | C (λ = 0.5) | R (λ = 0.5) | P (λ = 0.9) | C (λ = 0.9) | R (λ = 0.9) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ProMoRouter v2 (joint GNN) | 0.43822 ± 0.03229 | 0.15901 ± 0.09619 | 0.42232 ± 0.02837 | 0.42186 ± 0.02413 | 0.13774 ± 0.07168 | 0.35299 ± 0.04349 | 0.37433 ± 0.08145 | 0.11676 ± 0.08018 | 0.26925 ± 0.06614 |
| GraphRouter Direct | 0.35265 ± 0.15110 | 0.23778 ± 0.05778 | 0.32887 ± 0.14546 | 0.30444 ± 0.12337 | 0.16790 ± 0.07642 | 0.22049 ± 0.12132 | 0.30581 ± 0.12310 | 0.16767 ± 0.08079 | 0.15491 ± 0.12971 |
| Joint Flat-MLP | 0.56265 ± 0.05133 | 0.24855 ± 0.02924 | 0.53779 ± 0.05227 | 0.51463 ± 0.05447 | 0.14674 ± 0.03833 | 0.44126 ± 0.05530 | 0.47891 ± 0.06724 | 0.09440 ± 0.02208 | 0.39395 ± 0.04881 |
| Best fixed pair | 0.54541 ± 0.04092 | 0.25846 ± 0.01447 | 0.51957 ± 0.04123 | 0.54541 ± 0.04092 | 0.25846 ± 0.01447 | 0.41618 ± 0.04295 | 0.42961 ± 0.02253 | 0.07975 ± 0.00452 | 0.35783 ± 0.02173 |
| Best fixed model (Direct) | 0.48805 ± 0.05857 | 0.28844 ± 0.01128 | 0.45920 ± 0.05853 | 0.43943 ± 0.09096 | 0.22112 ± 0.08871 | 0.32887 ± 0.06264 | 0.37248 ± 0.07992 | 0.13139 ± 0.03587 | 0.25423 ± 0.05106 |
| Largest LLM (Direct) | 0.36355 ± 0.04497 | 0.39758 ± 0.01934 | 0.32380 ± 0.04557 | 0.36355 ± 0.04497 | 0.39758 ± 0.01934 | 0.16476 ± 0.04869 | 0.36355 ± 0.04497 | 0.39758 ± 0.01934 | 0.00573 ± 0.05277 |
| Smallest LLM (Direct) | 0.21036 ± 0.01547 | 0.05234 ± 0.00267 | 0.20512 ± 0.01566 | 0.21036 ± 0.01547 | 0.05234 ± 0.00267 | 0.18419 ± 0.01643 | 0.21036 ± 0.01547 | 0.05234 ± 0.00267 | 0.16325 ± 0.01724 |
| Joint oracle | 0.76826 ± 0.04210 | 0.11745 ± 0.00836 | 0.75651 ± 0.04168 | 0.76314 ± 0.04167 | 0.09925 ± 0.00762 | 0.71351 ± 0.04009 | 0.75774 ± 0.04225 | 0.09165 ± 0.00792 | 0.67526 ± 0.03853 |
| Direct oracle | 0.64079 ± 0.05873 | 0.17254 ± 0.01389 | 0.62354 ± 0.05886 | 0.63404 ± 0.05875 | 0.14917 ± 0.01066 | 0.55946 ± 0.05916 | 0.62502 ± 0.06266 | 0.13705 ± 0.01314 | 0.50168 ± 0.05889 |

Joint Flat-MLP achieves the highest mean reward among non-oracle methods in all three regimes. It exceeds GraphRouter Direct in 5/5 seeds and the best fixed pair in 4/5 seeds at each λ. The joint GNN exceeds GraphRouter Direct in 2/5, 5/5, and 5/5 seeds as λ increases, and ranks below Joint Flat-MLP in all 15 seed–regime comparisons. The joint methods score 36 prompt–model pairs, while GraphRouter Direct scores nine models under the Direct prompt. The joint oracle's mean reward exceeds the Direct oracle's by 0.13298, 0.15405, and 0.17358 across the three regimes.

I recalculated P, C, and R from the saved decisions and verified that all 28,692 cost proxies match the recorded token totals. Verifying that multi-call strategies capture every call remains pending.
