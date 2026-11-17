"""
Machine description and resource monitoring for the runs of the pipeline.

    machine_info()    what the experiments ran on: CPU, memory, GPUs, software versions and the SLURM allocation
    ResourceMonitor   samples CPU, memory and GPU use (and GPU power) of a process tree while it runs

scripts/run_stage.py uses both to wrap any command of the pipeline.
"""
import datetime
import importlib.metadata
import os
import platform
import shutil
import socket
import subprocess
import threading
import time

import psutil

PACKAGES = ('autogluon.tabular', 'torch', 'scikit-learn', 'numpy', 'pandas', 'ray', 'xgboost', 'psutil', 'pyserial')
SLURM_VARS = ('SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_JOB_PARTITION', 'SLURM_JOB_NODELIST', 'SLURM_CPUS_PER_TASK',
              'SLURM_CPUS_ON_NODE', 'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU', 'SLURM_NTASKS', 'SLURM_GPUS',
              'SLURM_GPUS_ON_NODE', 'SLURM_JOB_GPUS', 'SLURM_SUBMIT_DIR')


def _out(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def cpu_model():
    if os.path.exists('/proc/cpuinfo'):
        with open('/proc/cpuinfo', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.lower().startswith('model name'):
                    return line.split(':', 1)[1].strip()
    if platform.system() == 'Windows':
        name = _out(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor | Select-Object -First 1).Name'])
        if name:
            return name
    return platform.processor() or None


def gpu_info():
    out = _out(['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version,power.limit', '--format=csv,noheader,nounits'])
    gpus = []
    for line in (out or '').splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) == 5:
            gpus.append({'index': int(parts[0]), 'name': parts[1], 'memory_mib': _float(parts[2]), 'driver': parts[3],
                         'power_limit_w': _float(parts[4])})
    return gpus


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_info():
    try:
        import torch
    except ImportError:
        return None
    info = {'version': torch.__version__, 'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'intra_op_threads': torch.get_num_threads()}
    try:
        info['quantized_engine'] = torch.backends.quantized.engine
        info['supported_quantized_engines'] = list(torch.backends.quantized.supported_engines)
    except Exception:
        pass
    return info


def slurm_info():
    """Allocation of the current SLURM job (empty outside SLURM)."""
    env = {k: os.environ[k] for k in SLURM_VARS if k in os.environ}
    if 'SLURM_JOB_ID' in env:
        line = _out(['scontrol', 'show', 'job', '-o', env['SLURM_JOB_ID']])
        if line:
            fields = dict(tok.split('=', 1) for tok in line.split() if '=' in tok)
            for k in ('NumNodes', 'NumCPUs', 'TimeLimit', 'Partition', 'NodeList', 'TRES', 'Gres'):
                if k in fields:
                    env['scontrol_' + k] = fields[k]
    return env


def machine_info(include_node=True):
    vm = psutil.virtual_memory()
    freq = None
    try:
        f = psutil.cpu_freq()
        freq = round(f.max or f.current) if f else None
    except Exception:
        pass
    return {
        'timestamp_utc': utc_now(),
        'node': socket.gethostname() if include_node else None,
        'os': platform.platform(),
        'python': platform.python_version(),
        'cpu': {'model': cpu_model(), 'physical_cores': psutil.cpu_count(logical=False), 'logical_cpus': psutil.cpu_count(),
                'usable_cpus': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None, 'max_mhz': freq},
        'ram_total_gb': round(vm.total / 1e9, 1),
        'gpus': gpu_info(),
        'packages': package_versions(),
        'torch': torch_info(),
        'slurm': slurm_info(),
    }


class ResourceMonitor:
    """Samples a process tree (CPU, resident memory) and the GPUs (utilisation, memory, power) every `interval` s."""

    def __init__(self, pid, interval=1.0, gpu=True):
        self.root = psutil.Process(pid)
        self.interval = interval
        self.gpu = gpu and shutil.which('nvidia-smi') is not None
        self.samples = []
        self.t0 = self.t1 = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._known = {}

    def start(self):
        self.t0 = time.time()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)
        self.t1 = time.time()

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append(self._sample())
            self._stop.wait(self.interval)

    def _tree(self):
        try:
            procs = [self.root] + self.root.children(recursive=True)
        except psutil.Error:
            return []
        tree = []
        for p in procs:
            if p.pid not in self._known:
                self._known[p.pid] = p
                p.cpu_percent(None)   # the first call only primes the counter
            tree.append(self._known[p.pid])
        return tree

    def _sample(self):
        rss = cpu = 0.0
        pids = set()
        for p in self._tree():
            try:
                rss += p.memory_info().rss
                cpu += p.cpu_percent(None)
                pids.add(p.pid)
            except psutil.Error:
                pass
        s = {'t': time.time() - self.t0, 'rss_mb': rss / 1e6, 'cpu_pct': cpu, 'system_cpu_pct': psutil.cpu_percent(None)}
        if self.gpu:
            s.update(self._gpu_sample(pids))
        return s

    @staticmethod
    def _gpu_sample(pids):
        out = _out(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'], timeout=5)
        util, mem, power = [], [], []
        for line in (out or '').splitlines():
            u, m, w = [_float(x.strip()) for x in line.split(',')]
            util.append(u), mem.append(m), power.append(w)
        apps = _out(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], timeout=5)
        mine = None
        if apps is not None:
            mine = 0.0
            for line in apps.splitlines():
                try:
                    pid, used = line.split(',')
                    if int(pid) in pids and _float(used.strip()) is not None:
                        mine += _float(used.strip())
                except ValueError:
                    pass
        return {'gpu_util_pct': util, 'gpu_mem_mib': mem, 'gpu_power_w': power, 'gpu_proc_mem_mib': mine}

    def summary(self):
        s = self.samples
        wall = (self.t1 or time.time()) - self.t0
        out = {'wall_s': round(wall, 3), 'samples': len(s), 'interval_s': self.interval}
        if not s:
            return out
        rss = [x['rss_mb'] for x in s]
        cpu = [x['cpu_pct'] for x in s]
        out.update(peak_rss_mb=round(max(rss), 1), mean_rss_mb=round(sum(rss) / len(rss), 1),
                   mean_cpu_pct=round(sum(cpu) / len(cpu), 1), peak_cpu_pct=round(max(cpu), 1),
                   mean_system_cpu_pct=round(sum(x['system_cpu_pct'] for x in s) / len(s), 1),
                   cpu_time_s_estimated=round(sum(c / 100.0 for c in cpu) * self.interval, 1))
        if self.gpu and 'gpu_util_pct' in s[0]:
            gpus = []
            for g in range(len(s[0]['gpu_util_pct'])):
                util = [x['gpu_util_pct'][g] for x in s if g < len(x['gpu_util_pct']) and x['gpu_util_pct'][g] is not None]
                mem = [x['gpu_mem_mib'][g] for x in s if g < len(x['gpu_mem_mib']) and x['gpu_mem_mib'][g] is not None]
                pw = [(x['t'], x['gpu_power_w'][g]) for x in s if g < len(x['gpu_power_w']) and x['gpu_power_w'][g] is not None]
                energy = sum((pw[i][1] + pw[i - 1][1]) / 2 * (pw[i][0] - pw[i - 1][0]) for i in range(1, len(pw))) / 3600 if len(pw) > 1 else None
                gpus.append({'index': g, 'mean_util_pct': round(sum(util) / len(util), 1) if util else None,
                             'peak_mem_mib': max(mem) if mem else None,
                             'mean_power_w': round(sum(p for _, p in pw) / len(pw), 1) if pw else None,
                             'energy_wh_estimated': round(energy, 3) if energy is not None else None})
            mine = [x['gpu_proc_mem_mib'] for x in s if x.get('gpu_proc_mem_mib') is not None]
            out['gpus'] = gpus
            out['gpu_peak_mem_of_this_process_mib'] = max(mine) if mine else None
        return out
