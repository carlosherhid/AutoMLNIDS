"""Drives the benchmark builds (arduino_m5_ids_{int8,fp32}_bench) over the serial port.

  eval   sends rows of a raw Min-Max CSV (61 features + label, no header) and stores one line per sample:
         idx,pred,true,latency_us,rescale_us   (pred = -1 when the board did not answer)
         python run_device_eval.py eval --port <PORT> --input_csv ../data/test_device.csv --output_csv runs/int8_full.csv
         add --limit N for a subset, --resume to continue an interrupted run

  bench  runs the self-paced stream on the board (CPU load, deadline misses, throughput, heap, stack):
         python run_device_eval.py bench --port <PORT> --rates 10,50,100,125,1000 --seconds 30 --output_csv runs/int8_bench.csv

Besides the output CSV, eval writes <output>_stat.csv (heap and stack every 5000 samples) and <output>_summary.json
(wall time, answered samples, errors).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import serial  # pyserial


def open_port(port, baud):
    ser = serial.Serial(port, baud, timeout=2)
    time.sleep(2)  # the board resets when the port is opened
    ser.reset_input_buffer()
    deadline = time.time() + 15
    while time.time() < deadline:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print('DEVICE:', line)
        if 'READY' in line:
            return ser
    print('No READY received: press RESET on the M5Stack and run the script again (continuing anyway).', file=sys.stderr)
    return ser


def read_until(ser, prefix, timeout_s):
    """Returns (matching line or None, other lines read meanwhile)."""
    other, deadline = [], time.time() + timeout_s
    while time.time() < deadline:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue
        if line.startswith(prefix):
            return line, other
        other.append(line)
    return None, other


def run_eval(args):
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = args.start
    mode = 'w'
    if args.resume and out.exists():
        with open(out, encoding='utf-8') as f:
            start = max(start, sum(1 for _ in f) - 1)
        mode = 'a'
        print(f'resuming at row {start}')
    stat_path = out.with_name(out.stem + '_stat.csv')
    ser = open_port(args.port, args.baud)
    t_begin, answered, errors, sent = time.time(), 0, 0, 0
    with open(args.input_csv, encoding='utf-8') as fin, open(out, mode, newline='', encoding='utf-8') as fout, \
            open(stat_path, mode, newline='', encoding='utf-8') as fstat:
        w, ws = csv.writer(fout), csv.writer(fstat)
        if mode == 'w':
            w.writerow(['idx', 'pred', 'true', 'latency_us', 'rescale_us'])
            ws.writerow(['idx', 'free_heap', 'min_free_heap', 'heap_size', 'stack_free_bytes', 'n_samples'])
        for idx, line in enumerate(fin):
            if idx < start:
                continue
            if args.limit is not None and idx >= start + args.limit:
                break
            line = line.strip()
            if not line:
                continue
            ser.write((line + '\n').encode())
            resp, other = read_until(ser, 'PRED,', 5)
            for o in other:
                if o.startswith('STAT,'):
                    ws.writerow([idx] + o.split(',')[1:])
                elif o.startswith('ERR'):
                    errors += 1
            sent += 1
            if resp is None:
                errors += 1
                w.writerow([idx, -1, -1, -1, -1])
                ser.reset_input_buffer()
                print(f'[WARN] no answer for row {idx}')
            else:
                parts = resp.split(',')
                w.writerow([idx, int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]) if len(parts) > 4 else 0])
                answered += 1
            if sent % 1000 == 0:
                fout.flush(); fstat.flush()
                el = time.time() - t_begin
                print(f'{sent} rows sent | {answered} answered | {errors} errors | {el / 60:.1f} min | {sent / el:.1f} rows/s', flush=True)
    wall = time.time() - t_begin
    ser.close()
    summary = {'rows_sent': sent, 'rows_answered': answered, 'errors': errors, 'wall_seconds': round(wall, 1), 'baud': args.baud,
               'input_csv': str(args.input_csv)}
    out.with_name(out.stem + '_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('done:', summary)


def run_bench(args):
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    ser = open_port(args.port, args.baud)
    cols = ['rate_hz', 'seconds', 'n', 'misses', 'busy_us', 'elapsed_us', 'mean_us', 'max_us', 'rescale_mean_us', 'free_heap',
            'min_free_heap', 'stack_free_bytes']
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rate in [int(r) for r in args.rates.split(',')]:
            ser.write(f'BENCH,{rate},{args.seconds}\n'.encode())
            resp, _ = read_until(ser, 'BENCH,', args.seconds + 20)
            if resp is None:
                print(f'[WARN] no answer for rate {rate}')
                continue
            row = resp.split(',')[1:]
            w.writerow(row)
            f.flush()
            d = dict(zip(cols, row))
            load = float(d['busy_us']) / float(d['elapsed_us']) * 100
            print(f"rate {rate:5d}/s: {d['n']} inferences, {d['misses']} deadline misses, CPU load {load:.1f}%, "
                  f"throughput {int(d['n']) / (float(d['elapsed_us']) / 1e6):.1f}/s, mean {d['mean_us']} us, max {d['max_us']} us")
    ser.write(b'STAT\n')
    resp, _ = read_until(ser, 'STAT,', 5)
    print('final', resp)
    ser.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('eval')
    p.add_argument('--port', required=True)
    p.add_argument('--baud', type=int, default=921600)
    p.add_argument('--input_csv', required=True)
    p.add_argument('--output_csv', required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--resume', action='store_true')
    p.set_defaults(func=run_eval)
    p = sub.add_parser('bench')
    p.add_argument('--port', required=True)
    p.add_argument('--baud', type=int, default=921600)
    p.add_argument('--rates', default='10,50,100,125,1000')
    p.add_argument('--seconds', type=int, default=30)
    p.add_argument('--output_csv', required=True)
    p.set_defaults(func=run_bench)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
