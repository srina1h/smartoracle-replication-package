# SmartOracle replication package

Code and data for the SmartOracle study.

## Contents

- `smartoracle/`: SmartOracle implementation and prompts
- `baseline/`: sequential baseline and saved timing summaries
- `data/`: manual labels and saved feature vectors
- `analysis/clustering/`: root-cause propagation analysis
- `analysis/reasoning/`: analysis of 234 reasoning traces

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
./reproduce.sh
```

`reproduce.sh` checks the retained data and reruns the clustering and reasoning
analyses. API keys are not needed for these analyses.

The case-level manifest for the 374-item SmartOracle evaluation was not
retained. The paper reports the aggregate confusion matrix, but that evaluation
cannot be rerun case by case from this repository. See
`EVALUATION_SUBSET_STATUS.md`.

MIT License. Repository:
https://github.com/srina1h/smartoracle-replication-package
