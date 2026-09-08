# Sequential baseline

`baseline_runner.py` contains the complete sequential baseline prompts and runs
one case. `baseline_batch.py` applies the runner to a directory tree. Both call
the model API and are not part of the offline verification command.

Set `GOOGLE_API_KEY`, `BASELINE_INPUT_ROOT`, and `BASELINE_OUTPUT_ROOT` before
running the batch. The retained JSON summaries in `results/` record the runs
used for the paper's runtime comparison.

Example:

```bash
python3 baseline_runner.py --case path/to/case_data.json
```
