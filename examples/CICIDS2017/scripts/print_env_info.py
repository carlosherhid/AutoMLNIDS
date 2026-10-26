"""
Print the software/hardware environment of the machine it runs on, for the paper's
experimental-setup paragraph (library versions, CPU model, torch threading and
quantization backend). Run it on the same compute node as the latency measurements:

    sbatch --job-name=envinfo --time=00:05:00 --mem=4G --cpus-per-task=2 \
        code/cluster/run_python.sbatch code/examples/CICIDS2017/scripts/print_env_info.py
"""
import platform
import subprocess

import autogluon.tabular as autogluon_tabular
import sklearn
import torch

try:
    import xgboost
    xgboost_version = xgboost.__version__
except ImportError:
    xgboost_version = 'not installed'

print(f'node: {platform.node()}')
print(f'python {platform.python_version()} | torch {torch.__version__} | scikit-learn {sklearn.__version__} '
      f'| autogluon.tabular {autogluon_tabular.__version__} | xgboost {xgboost_version}')
print(f'torch intra-op threads: {torch.get_num_threads()} | quantized engine: {torch.backends.quantized.engine} '
      f'| supported engines: {torch.backends.quantized.supported_engines}')

wanted = ('Architecture', 'Model name', 'CPU(s):', 'Thread(s) per core', 'Core(s) per socket', 'Socket(s)')
for line in subprocess.run(['lscpu'], capture_output=True, text=True).stdout.splitlines():
    if line.startswith(wanted):
        print(line)
