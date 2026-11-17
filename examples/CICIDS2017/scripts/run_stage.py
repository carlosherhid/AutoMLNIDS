"""
Run one command of the pipeline and record where and how it ran: the machine, the wall-clock time, the CPU, memory and
GPU use (and an estimate of the GPU energy), and the exit code.

Usage:
    python scripts/run_stage.py --name <stage_name> --out_dir <telemetry_dir> -- <command and arguments>

For example:
    python scripts/run_stage.py --name qat_exp4_rep0 --out_dir telemetry -- \
        python scripts/autogluon_to_pytorch_qat.py --predictor_path <predictor_dir> --data_dir <data_dir> ...

Writes <out_dir>/<name>.json (full detail), appends one row to <out_dir>/stages.csv (one line per stage, ready for a table)
and, the first time, <out_dir>/machine.json. The exit code of the command is the exit code of this script.
On a SLURM cluster, cluster/run_python.sbatch calls it automatically when TELEMETRY_DIR is set.
"""
import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from utils.telemetry import ResourceMonitor, machine_info, slurm_info, utc_now  # noqa: E402

CSV_FIELDS = ['name', 'started_utc', 'wall_s', 'exit_code', 'cpu_time_s', 'peak_rss_mb', 'mean_cpu_pct', 'gpu_mean_util_pct',
              'gpu_peak_mem_mib', 'gpu_mean_power_w', 'gpu_energy_wh_estimated', 'node', 'slurm_job_id']


def main():
    parser = argparse.ArgumentParser(description='Run a command and record machine and resource use',
                                     usage='run_stage.py --name NAME [--out_dir DIR] [--interval S] [--no_node] [--tag k=v ...] -- COMMAND ...')
    parser.add_argument('--name', required=True, help='stage name; also the name of the JSON file')
    parser.add_argument('--out_dir', default='telemetry')
    parser.add_argument('--interval', type=float, default=1.0, help='sampling period in seconds')
    parser.add_argument('--no_node', action='store_true', help='do not record the host name')
    parser.add_argument('--tag', action='append', default=[], help='extra key=value stored with the stage (dataset, repetition, ...)')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == '--' else args.command
    if not command:
        parser.error('give the command after --')
    os.makedirs(args.out_dir, exist_ok=True)

    machine_path = os.path.join(args.out_dir, 'machine.json')
    if not os.path.exists(machine_path):
        with open(machine_path, 'w', encoding='utf-8') as f:
            json.dump(machine_info(include_node=not args.no_node), f, indent=2)

    started = utc_now()
    ru0 = None
    try:
        import resource
        ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    except ImportError:
        resource = None
    t0 = time.time()
    try:
        proc = subprocess.Popen(command)
    except OSError as e:
        print(f'[telemetry] cannot start {command[0]}: {e}', file=sys.stderr)
        sys.exit(127)
    monitor = ResourceMonitor(proc.pid, interval=args.interval)
    monitor.start()
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        code = proc.wait()
    monitor.stop()
    wall = time.time() - t0

    res = monitor.summary()
    if resource is not None:   # exact CPU time and largest resident set of the processes of the command (Linux)
        ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        res['cpu_user_s'] = round(ru1.ru_utime - ru0.ru_utime, 2)
        res['cpu_system_s'] = round(ru1.ru_stime - ru0.ru_stime, 2)
        res['max_rss_mb_rusage'] = round(ru1.ru_maxrss / 1024, 1)
    cpu_time = round(res['cpu_user_s'] + res['cpu_system_s'], 1) if 'cpu_user_s' in res else res.get('cpu_time_s_estimated')

    record = {'name': args.name, 'command': command, 'cwd': os.getcwd(), 'started_utc': started, 'finished_utc': utc_now(),
              'wall_s': round(wall, 3), 'exit_code': code, 'tags': dict(t.split('=', 1) for t in args.tag if '=' in t),
              'node': None if args.no_node else socket.gethostname(),
              'resources': res, 'slurm': slurm_info()}
    with open(os.path.join(args.out_dir, args.name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2)

    g = (res.get('gpus') or [{}])[0]
    row = {'name': args.name, 'started_utc': started, 'wall_s': record['wall_s'], 'exit_code': code, 'cpu_time_s': cpu_time,
           'peak_rss_mb': res.get('peak_rss_mb'), 'mean_cpu_pct': res.get('mean_cpu_pct'),
           'gpu_mean_util_pct': g.get('mean_util_pct'), 'gpu_peak_mem_mib': g.get('peak_mem_mib'),
           'gpu_mean_power_w': g.get('mean_power_w'), 'gpu_energy_wh_estimated': g.get('energy_wh_estimated'),
           'node': record['node'], 'slurm_job_id': record['slurm'].get('SLURM_JOB_ID')}
    csv_path = os.path.join(args.out_dir, 'stages.csv')
    new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)

    gpu_txt = f", GPU util {g['mean_util_pct']} % (peak {g['peak_mem_mib']} MiB)" if g.get('mean_util_pct') is not None else ''
    print(f"[telemetry] {args.name}: {record['wall_s']} s, peak RSS {res.get('peak_rss_mb')} MB, mean CPU {res.get('mean_cpu_pct')} %"
          f"{gpu_txt}, exit {code} -> {args.out_dir}", file=sys.stderr)
    sys.exit(code)


if __name__ == '__main__':
    main()
