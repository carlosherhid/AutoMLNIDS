# Patches

Two unrelated workarounds needed to run `automl_model_search.py` with the AutoGluon version pinned in
[`requirements.txt`](../../../../requirements.txt).

## 1. `apply_autogluon_patch.py`: one-expression fix of an AutoGluon file

**Not part of the pipeline: run it once per environment** (`cluster/setup_env.sbatch` does it after every installation). In some AutoGluon versions
`tabular/trainer/auto_trainer.py` logs `hyperparameters[k][:3]` for every model family, but the value of `NN_TORCH` is a dict (a search space),
not a list, so the search stops with `TypeError: unhashable type: 'slice'`. The script guards that one expression with an `isinstance(..., list)`
check. **AutoGluon 1.6.2, the pinned version, already has that guard, so there the script prints `already patched` and changes nothing**; it is
kept as a safety net for versions without it. It edits only that expression, never replaces the file (the file differs between AutoGluon
versions), refuses to write a file that does not parse, and can be run again safely:

```bash
python scripts/patches/apply_autogluon_patch.py
```

Earlier versions of this folder also held `autogluon_auto_trainer.py`, a full copy of that file taken from an old AutoGluon. It was removed: copied over
the installed file it makes `import autogluon.tabular` fail on AutoGluon 1.6.2 (`cannot import name 'MODEL_TYPES'`).

## 2. `sklearn_writeable_array_fix.py` — importable Python patch, not a file replacement

This one **is** meant to be imported (already wired into
[`automl_model_search.py`](../automl_model_search.py)'s own imports — nothing to
do by hand). It works around a `ValueError: cannot set WRITEABLE flag to True of
this array` that scikit-learn's `type_of_target()` hits internally on the
validation-label array AutoGluon's HPO trials pass it (root-caused: that array's
buffer isn't writeable, likely from AutoGluon's HPO data-pickling round-trip -
see the file's own docstring for the full chain). `type_of_target()` silently
*catches* that ValueError and falls back to a `dtype=object` array, which then
gets misclassified as `"unknown"` instead of `"binary"` — surfacing as
`ValueError: Classification metrics can't handle a mix of unknown and binary
targets` on every single HPO trial, 100% reproducible regardless of
hyperparameters. Confirmed present on both AutoGluon 1.0.0 and 1.6.2 (so it's a
scikit-learn/data-pipeline interaction, not an AutoGluon version regression);
the docstring of that file has the full diagnostic trail.

If you add another script that calls into AutoGluon's HPO tuning (only
`automl_model_search.py` does today), add near its top:

```python
import patches.sklearn_writeable_array_fix  # noqa: F401
```
