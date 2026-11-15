# Results summary

## Full SD 1.4 matrix

The full three-target, nine-condition matrix has **NOT MEASURED** ASR in this checkout. The table below lists reference values encoded in `PAPER_ASR` and confirmed against [Table 4 of Braun et al.](https://arxiv.org/html/2504.21072v4), not results from this project. This implementation also differs in templates, target choices, and COCO sample size; a numerical match would not establish an exact reproduction.

| Erasure | Attack | Paper reference ASR (%) | Full-matrix ASR |
| --- | --- | ---: | --- |
| UCE | none | 10.0 | NOT MEASURED |
| UCE | shallow | 92.0 | NOT MEASURED |
| UCE | deep | 94.2 | NOT MEASURED |
| ESD | none | 10.0 | NOT MEASURED |
| ESD | shallow | 38.0 | NOT MEASURED |
| ESD | deep | 70.8 | NOT MEASURED |

The prior 2026-11-10 bird feasibility run at one image per prompt found 0/1 trigger classifications for clean, no-attack UCE, and shallow-plus-UCE. The explicit bird prompt was classified bird in all three. These observations cannot estimate trigger persistence: the tested shallow trigger did not succeed before or after erasure, and the UCE output was still classified bird. The run used 512×512 pixels, 20 DDIM steps, seed 42, guidance 7.5, and a CPU. Its artifacts are described in `docs/verification.md`; they are absent from this checkout unless restored from the release.

## Completed CPU medium pilot, 2026-11-15

The bird-only SD 1.4 pilot completed all nine conditions on CPU with seed 42
(image seeds 42–46), five explicit-target and five trigger images per condition,
256×256 pixels, 10 DDIM steps, guidance 7.5, ten EEBdeep optimization steps,
and ten ESD optimization steps. Values below are classifier counts from
`results/sd14_cpu_medium/tables/`, with percentages in parentheses. The paper
column is the reference value encoded in this repository; comparisons are
descriptive because the sample sizes and training settings differ.

| Attack | Erasure | Target classified bird | Trigger classified bird | Paper reference ASR (%) |
| --- | --- | ---: | ---: | ---: |
| none | clean | 2/5 (40%) | 1/5 (20%) | — |
| none | ESD | 2/5 (40%) | 1/5 (20%) | 10.0 |
| none | UCE | 1/5 (20%) | 0/5 (0%) | 10.0 |
| shallow | clean | 1/5 (20%) | 2/5 (40%) | — |
| shallow | ESD | 2/5 (40%) | 2/5 (40%) | 38.0 |
| shallow | UCE | 1/5 (20%) | 3/5 (60%) | 92.0 |
| deep | clean | 2/5 (40%) | 1/5 (20%) | — |
| deep | ESD | 2/5 (40%) | 1/5 (20%) | 70.8 |
| deep | UCE | 2/5 (40%) | 0/5 (0%) | 94.2 |

The shallow trigger count remained 2/5 after ESD and was 3/5 after UCE,
versus 2/5 before erasure. The deep trigger count was 1/5 before erasure,
the same as the no-attack clean control; it remained 1/5 after ESD and was
0/5 after UCE. These observations do **not** establish persistence: the
pre-erasure attacks were weak or absent at these settings, and explicit-target
accuracy of 20–40% does not validate effective erasure. Each trigger count
changes by 20 percentage points per image. Only one target was measured, so
there is no across-target sample standard deviation. Retention accuracy, FID,
and CLIPScore remain **NOT MEASURED** for this pilot.

## Limitations

- Phan's CIFAR-10 classifier assigns one of ten coarse classes to each generated image. It can miss target content and can label unusual images incorrectly.
- FID from 100 COCO 2014 image/caption pairs is noisy and is not directly comparable with a larger reference set.
- The medium pilot uses reduced image size, sampling steps, and optimization steps. It is unsuitable for a numerical comparison with the paper.
- The full method uses ten declared EEBdeep templates rather than the paper's reported 80 templates; the latter list is unavailable in the linked code. Ship and dog are configurable target choices.
- A trigger persistence claim requires both a successful pre-erasure attack and a successful erasure check, in addition to post-erasure ASR. Neither has been established by the prior one-image run.
