"""
Fixes, in place, the `unhashable type: 'slice'` error that AutoGluon raises while logging the hyperparameters of the search.

AutoGluon's `tabular/trainer/auto_trainer.py` logs `hyperparameters[k][:3]` for every model family, but the value of `NN_TORCH` is a
dict (a search space), not a list, and a dict cannot be sliced. The fix guards that one expression with `isinstance(..., list)`.

It edits only that expression of the installed file (it does not replace the file, which differs between AutoGluon versions), and it is
safe to run again. `cluster/setup_env.sbatch` runs it after every installation, on a pristine copy of the file.

Usage (inside the environment):
    python scripts/patches/apply_autogluon_patch.py [path/to/auto_trainer.py]
"""
import ast
import importlib.util
import os
import sys

OLD = '{hyperparameters[k][:3]}'
NEW = '{hyperparameters[k][:3] if isinstance(hyperparameters[k], list) else hyperparameters[k]}'
GUARD = 'isinstance(hyperparameters[k], list)'


def locate():
    # find the module without importing it: importing autogluon.tabular would run the code we are about to patch
    spec = importlib.util.find_spec('autogluon.tabular')
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit('autogluon.tabular is not installed in this environment')
    return os.path.join(list(spec.submodule_search_locations)[0], 'trainer', 'auto_trainer.py')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else locate()
    with open(path, encoding='utf-8') as f:
        source = f.read()
    if GUARD in source:
        print(f'{path}: already patched')
        return
    if source.count(OLD) != 1:
        # nothing to fix (a version without that expression), or a version that logs it in another way: leave the file as it is
        print(f'{path}: expression {OLD} found {source.count(OLD)} times; the file was not modified')
        return
    patched = source.replace(OLD, NEW)
    ast.parse(patched)          # never write a file that does not parse
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(patched)
    print(f'Patched {path}')


if __name__ == '__main__':
    sys.exit(main())
