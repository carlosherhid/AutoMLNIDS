# Device benchmark on the M5Stack Basic (ESP32)

Extends the on-device evaluation of Section 5.11 of the paper (single checkpoint, 100 and 1000 samples, mean latency,
rescaling done on the host) with: the **full test set** on the board, the **latency distribution and stability**,
**RAM** (heap and stack) at runtime, **CPU load and throughput** without serial traffic, the **cost of the near-zero
rescaling on the board**, and optionally the **five seeds** and the **energy per inference**.

## Files

| file | what it is |
|---|---|
| `bench_common.h` | shared firmware code: serial protocol, on-board rescaling, `BENCH` stream, heap/stack report |
| `arduino_m5_ids_int8_bench/`, `arduino_m5_ids_fp32_bench/` | the two sketches (same arithmetic as the validated ones) |
| `make_bench_samples.py` | writes `bench_samples.h` and copies the headers into the two sketch folders |
| `reference_predictions.py` | FP32 (PyTorch) and INT8 (integer emulation) reference predictions on the full test set |
| `run_device_eval.py` | `eval` (rows over the serial port) and `bench` (self-paced stream) |
| `analyze_device_run.py` | metrics, latency distribution, drift and agreement with the reference of one run |

The firmware receives the **raw Min-Max** rows (`data/test_device.csv`) and applies Equation 2 itself.

## Steps (run from `m5stack_ids_eval/`)

1. `python device_bench/make_bench_samples.py` and `python device_bench/reference_predictions.py`
   (the second one must print 99.53 / 99.75 / 0.70 for FP32 and 99.26 / 99.23 / 0.71 for INT8).
2. Open `device_bench/arduino_m5_ids_int8_bench/arduino_m5_ids_int8_bench.ino` in the Arduino IDE (same board settings
   as before), upload, and note the "program storage" and "global variables" lines of the compilation (they include
   24 KB of `bench_samples.h`; the footprint of Table 10 comes from the sketches without it).
3. Check with a subset: `python device_bench/run_device_eval.py eval --port <PORT> --input_csv data/test_device.csv --limit 1000 --output_csv device_bench/runs/int8_1000.csv`
   then `python device_bench/analyze_device_run.py --preds device_bench/runs/int8_1000.csv --ref device_bench/ref/ref_int8.csv`:
   it must report 0 different predictions and a latency of about 7.9 ms.
4. Full test set (about 1 hour for INT8 and 1.5 hours for FP32 at 921600 baud; keep the computer awake):
   `python device_bench/run_device_eval.py eval --port <PORT> --input_csv data/test_device.csv --output_csv device_bench/runs/int8_full.csv`
   (`--resume` continues an interrupted run), then `analyze_device_run.py` with `--ref device_bench/ref/ref_int8.csv`.
5. CPU load, throughput and deadline misses: `python device_bench/run_device_eval.py bench --port <PORT> --rates 10,50,100,125,150,1000 --seconds 30 --output_csv device_bench/runs/int8_bench.csv`.
6. Repeat 2-5 with `arduino_m5_ids_fp32_bench` (and `ref_fp32.csv`, rates 10,25,40,50,1000).

## Reading the results

* `latency_us` includes the rescaling and the input quantization; `rescale_us` is the rescaling alone.
* `BENCH`: CPU load = `busy_us / elapsed_us` at the given arrival rate; at a rate above the capacity the number of
  inferences per second is the maximum throughput and `misses` grows. The board spins while waiting for the next sample.
* RAM: `STAT` gives the free heap, its minimum, and the free stack of the Arduino task in bytes; static data is the
  "global variables" figure of the compiler.

## Optional extensions

* **Five seeds:** export the 5 `qat_pytorch_model_exp1006XX.pth` / `normal_pytorch_model_exp1006XX.pth` checkpoints of the cluster
  with `export_qat_int8.py` and `export_mlp_to_header.py`, rebuild the headers and repeat step 4 with `--limit 20000`.
* **Energy:** needs an inline USB power meter with logging (or an INA219). Run the `BENCH` stream for 60 s at the maximum rate
  for FP32 and INT8, and an idle baseline with the display off; energy per inference = (P_active - P_idle) / throughput.
