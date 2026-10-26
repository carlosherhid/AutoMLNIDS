# Patches

Two unrelated workarounds needed to run `automl_model_search.py` with the AutoGluon version pinned in
[`requirements.txt`](../../../../requirements.txt).

## 1. `autogluon_auto_trainer.py` — AutoGluon library file replacement

**Not a script to run.** It is a drop-in replacement for AutoGluon's own internal
file:

```
<your-venv>/lib/python3.X/site-packages/autogluon/tabular/trainer/auto_trainer.py
```

It works around an `unhashable type: 'slice'` error that AutoGluon raises during
hyperparameter search on this project's setup (`fit()` logs hyperparameters by
slicing `hyperparameters[k][:3]` unconditionally, but `NN_TORCH`'s value is a
dict, not a list — this patch adds an `isinstance(..., list)` check first). To
apply it by hand, copy this file over the one installed in your virtual
environment (back up the original first):

```bash
cp scripts/patches/autogluon_auto_trainer.py <your-venv>/lib/python3.X/site-packages/autogluon/tabular/trainer/auto_trainer.py
```

Confirmed byte-identical to the installed file (aside from the fix above) on
AutoGluon 0.8.2, 1.0.0, and 1.6.2 (checked by diffing against the actual installed
file each time). If a newer AutoGluon version ever changes this
file further, `cp` may need to go back to being a smaller hand-applied diff
instead of a full-file replacement.

[`cluster/setup_env.sbatch`](../../../../cluster/setup_env.sbatch) applies this
patch automatically after every `pip install`, so on the cluster you don't need
to run the command above by hand.

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
