# V19 three-track experiment protocol

## Independent tracks

| Track | Train | Validation | Sealed test |
|---|---:|---:|---:|
| MARBLE-only | 4,565 | 1,791 | 1,491 |
| AutoGen-only | 1,023 | 425 | 337 |
| Mixed | 5,588 | 2,216 | 1,828 |

Each track starts independently from the same Qwen3-8B base model. No adapter is
continued from another track. Hyperparameters and seed are identical.

## Validation protocol

Run the unmodified validation set plus seven counterfactuals:

1. task goal masked;
2. all event text masked;
3. final outcome text masked;
4. topology/edges/candidate-event links masked;
5. top train-derived TF-IDF cues masked;
6. observable event order shuffled;
7. event text rotated from a different-verdict validation sample.

The lexical mask is learned from training text only. Validation labels are not
used to select lexical features. No counterfactual is generated from final test.

Interpretation is based on delta from clean validation. A large text-mask or
lexical-mask drop indicates textual shortcut dependence. A large structure-mask
drop indicates graph dependence. A large order-shuffle drop indicates temporal
reasoning. A large outcome-mask drop indicates outcome verification. Performance
on cross-label text rotation reveals whether predictions follow structure or
unrelated text.

## Final reporting

After all choices are frozen, evaluate each track's own sealed test once. Report
all three result rows separately. The average version is the unweighted macro
mean and population standard deviation across the three rows; it is not a fourth
model and must not replace the individual results.
