"""
Fix + diagnostics for a 'Classification metrics can't handle a mix of unknown and
binary targets' crash hit during AutoGluon's NN_TORCH hyperparameter search on a
SLURM cluster setup (confirmed on both AutoGluon 1.0.0 and 1.6.2, both the
ray-tune and custom-hpo-logic backends).

Round 1 (insufficient on its own): probing found scikit-learn's
`type_of_target()` internally calls `check_array()`, which raised `ValueError:
cannot set WRITEABLE flag to True of this array` on the array probed at the time;
`type_of_target()` catches that ValueError internally and falls back to
`dtype=object`, which then gets misclassified as "unknown" instead of "binary".
The fix applied for that - pre-copying y_true/y_pred to writeable numpy arrays
before sklearn's `_check_targets()` ever sees them - is still applied below
(`_make_writeable` / `_patched_check_targets`), but a full smoke-test run
showed it does NOT actually resolve the crash: `_patched_check_targets`
runs (confirmed in the traceback), yet `_orig_check_targets` still raises the same
error. So the array that broke during standalone probing isn't necessarily the
same array (or the same failure mode) hit during a real HPO trial.

Round 2 (diagnostics added): wrapped `type_of_target` itself (the name
as imported into sklearn.metrics._classification's own namespace, same trick as
`_check_targets` below - Python resolves globals at call time via
`func.__globals__`, so patching the module attribute here intercepts calls made by
the original `_check_targets`/`accuracy_score` too) so that whenever it is about
to return "unknown" - or raise - it dumps to stderr: the input's python type,
numpy dtype, shape, writeable flag, a sample of raw values and their python types,
and (by re-invoking check_array standalone) the actual internal exception sklearn
silently swallows. Ran this on the cluster and got a clean answer: it's `y_true`
(a uint8 Series, clean data, no NaN/None), reported `writeable: True` by our own
probe, yet `check_array()` on that same object still raised the WRITEABLE-flag
ValueError immediately after.

Root cause (confirmed): pandas 2.x Copy-on-Write. A Series extracted from a
DataFrame column can report `writeable=True` the instant you check it, then flip
to a protected read-only view of the same underlying block once pandas' internal
reference tracking notices another view exists (e.g. AutoGluon holding its own
reference, or even our own diagnostics taking a second view) - so the flag is a
snapshot, not a guarantee. Round 1's `_make_writeable` only copied when the flag
was *already* False at check time, which missed this: the flag can still be True
when we check, then be False (and backed by a read-only base) by the time
`check_array` gets to it moments later. Round 2's `_make_writeable` fixes this by
copying unconditionally (`Series.to_numpy(copy=True)` / `np.array(y, copy=True)`)
instead of gating on the current flag value.

Diagnostics stay in place (read the job's .err file for lines starting with
"[sklearn_writeable_array_fix]") in case a different array trips the same
"unknown" fallback in the future.

Usage: `import patches.sklearn_writeable_array_fix` (or add scripts/patches to
sys.path first) near the top of any script that calls into AutoGluon's scoring -
currently only automl_model_search.py exercises the affected code path (HPO
tuning). Safe to import multiple times (idempotent).
"""
import sys

import numpy as np
import pandas as pd
import sklearn.metrics._classification as _classification_mod

_PATCHED_MARKER = "_writeable_array_fix_applied"


def _make_writeable(y):
    """Always force a real copy, regardless of the array's *current* writeable
    flag. Round 1 of this fix only copied when `flags.writeable` was already
    False, which looked right but wasn't: under pandas 2.x Copy-on-Write, a
    Series extracted from a DataFrame column can report writeable=True at the
    moment we check it, then flip to a protected read-only view of the same
    block by the time something downstream (sklearn's own check_array, several
    calls later) takes its own view - confirmed via the diagnostics
    below: the array reported writeable=True right before check_array() still
    raised 'cannot set WRITEABLE flag to True of this array' on it. Since the
    flag is unreliable, always copy instead of conditionally copying."""
    if isinstance(y, (pd.Series, pd.Index)):
        return y.to_numpy(copy=True)
    try:
        return np.array(y, copy=True)
    except Exception:
        return y


def _describe(y):
    info = {"python_type": type(y).__name__}
    if isinstance(y, pd.Series):
        info["pandas_dtype"] = str(y.dtype)
    try:
        arr = np.asarray(y)
        info["dtype"] = str(arr.dtype)
        info["shape"] = arr.shape
        info["writeable"] = bool(arr.flags.writeable) if hasattr(arr, "flags") else None
        flat = arr.reshape(-1)
        head = flat[:10].tolist()
        info["sample"] = head
        info["sample_python_types"] = sorted({type(v).__name__ for v in flat[:50].tolist()})
        info["n_none_or_nan_in_sample"] = sum(
            1 for v in flat[:200].tolist()
            if v is None or (isinstance(v, float) and v != v)
        )
    except Exception as e:
        info["describe_error"] = repr(e)
    return info


def _probe_check_array(y):
    """Re-run sklearn's own internal check_array() call standalone to surface the
    exact exception type_of_target() silently swallows. Tries both the old
    (force_all_finite) and new (ensure_all_finite) kwarg names since this project
    has hit sklearn version differences before."""
    from sklearn.utils import check_array
    common_kwargs = dict(dtype=None, ensure_2d=False, ensure_min_samples=0)
    for finite_kwarg in ("ensure_all_finite", "force_all_finite"):
        try:
            check_array(y, **common_kwargs, **{finite_kwarg: False})
            return None  # succeeded - the real crash's check_array call differs from this one
        except TypeError:
            continue  # wrong kwarg name for this sklearn version, try the other
        except Exception as e:
            return e
    try:
        check_array(y, **common_kwargs)
        return None
    except Exception as e:
        return e


def _dump_diagnostic(tag, y, exc=None):
    print(f"[sklearn_writeable_array_fix] type_of_target({tag!r}) -> 'unknown' | "
          f"info={_describe(y)}"
          + (f" | internal_exc={exc!r}" if exc is not None else ""),
          file=sys.stderr, flush=True)
    probe_exc = _probe_check_array(y)
    if probe_exc is None:
        print(f"[sklearn_writeable_array_fix] standalone check_array({tag!r}) "
              f"succeeded - could not reproduce the internal failure directly",
              file=sys.stderr, flush=True)
    else:
        print(f"[sklearn_writeable_array_fix] standalone check_array({tag!r}) "
              f"RAISED: {probe_exc!r}", file=sys.stderr, flush=True)


if not getattr(_classification_mod, _PATCHED_MARKER, False):
    _orig_check_targets = _classification_mod._check_targets
    _orig_type_of_target = _classification_mod.type_of_target

    def _patched_check_targets(y_true, y_pred):
        return _orig_check_targets(_make_writeable(y_true), _make_writeable(y_pred))

    def _diagnostic_type_of_target(*args, **kwargs):
        tag = kwargs.get("input_name") or "y"
        y = args[0] if args else kwargs.get("y")
        try:
            result = _orig_type_of_target(*args, **kwargs)
        except Exception as e:
            _dump_diagnostic(tag, y, exc=e)
            raise
        if result == "unknown":
            _dump_diagnostic(tag, y)
        return result

    _classification_mod._check_targets = _patched_check_targets
    _classification_mod.type_of_target = _diagnostic_type_of_target
    setattr(_classification_mod, _PATCHED_MARKER, True)
