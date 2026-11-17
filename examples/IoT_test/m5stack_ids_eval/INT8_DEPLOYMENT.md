# INT8 (QAT) deployment on the M5Stack Basic

The original firmware (`arduino_m5_ids_normal_exp6`) runs the **FP32** model (float32 weights, float
arithmetic). `arduino_m5_ids_int8` runs the **QAT-converted** model with integer arithmetic.

## What is exported

`export_qat_int8.py` reads `qat_pytorch_model_exp6.pth` (the QAT counterpart of `normal_pytorch_model_exp6.pth`,
same training run) and writes `arduino_m5_ids_int8/exp6_qat_int8_model.h`:

* `int8_t` weights (50,524 bytes) and `int32_t` biases (3,000 bytes) of the 12 Linear layers;
* the input quantization (scale, zero point), and per layer the input/output zero points and the float
  requantization multiplier `s_in * s_w / s_out`.

The checkpoint does not record its activation function and the firmware implements only ReLU, so the exporter stops unless it is told what the
network is: `--architecture_json <automl_model_summary.json>` (written by `describe_automl_model.py`) checks that the network has no layer the
firmware lacks (the AutoML search can choose Tanh) and that its Linear layers are those of the checkpoint (the width of the first layer's input is
not compared: the AutoGluon network can take more inputs than the data has columns); `--assume_relu` skips the check for a network known to use ReLU
(the checkpoint included in this folder is one). `export_mlp_to_header.py`, for the FP32 firmware, takes the same two options. The firmware also
limits the network to 64 inputs and 128 neurons per layer (`MAX_INPUTS`, `MAX_NEURONS`); the exporters do not check it.

Before flashing, the script checks that the exact integer arithmetic of the firmware reproduces PyTorch's
quantized model (engine `onednn`, which does not saturate), and that the header, parsed back, is the same
network. Results with this checkpoint (rescaled CICIDS2017 test set, 166,967 rows):

| model | accuracy | recall | FPR |
|---|---|---|---|
| FP32 (`normal_pytorch_model_exp6.pth`) | 99.53% | 99.75% | 0.70% |
| QAT, PyTorch quantized (onednn) | 99.26% | 99.23% | 0.71% |
| QAT, integer arithmetic of the firmware | 99.26% | 99.23% | 0.71% |

The integer emulation and PyTorch disagree on 3 of 166,967 rows. The differences are rounding effects, mainly of the bias: PyTorch's `onednn` engine adds
it in float, while the firmware stores it as an int32 rounded to the accumulator scale, so a few rows can differ by one output step; the effect
grows with the depth of the network.

Note: these figures belong to the checkpoint included in this folder (CICIDS2017, 61 inputs), which was trained with an earlier version of the data
pipeline. The exporter, the firmware and the checks do not depend on the dataset: with another checkpoint, `export_qat_int8.py --check_csv` prints
its own figures.

## Run it

1. Copy `arduino_m5_ids_int8/` (sketch + header) next to `arduino_m5_ids_normal_exp6/`.
2. Open `arduino_m5_ids_int8/arduino_m5_ids_int8.ino` in the Arduino IDE (same board and settings as the FP32
   sketch), compile and upload. Keep the lines that report program storage and global variables.
3. With the board connected, from the `m5stack_ids_eval` folder:

```
python run_serial_eval.py --port <PORT> --input_csv data/test_device_scaled_100.csv  --output_csv preds_m5_exp6_int8_100.csv
python run_serial_eval.py --port <PORT> --input_csv data/test_device_scaled_1000.csv --output_csv preds_m5_exp6_int8_1000.csv
```

The full test set, with accuracy, recall, FPR and the latency distribution, is evaluated with the scripts of
`device_bench/` (see its README).

The `test_device_scaled_*` files are the same samples used for the FP32 firmware (features already rescaled on the
host, label in the last column), so both models see identical inputs.

NOTE: the note on QAT engines. Do not use `--engine` values other than `onednn` for the reference check on a CPU
without VNNI: the default `x86` engine falls back to fbgemm there and saturates (a QAT checkpoint dropped from
99.4% to 74.9% in a test on a desktop CPU).
