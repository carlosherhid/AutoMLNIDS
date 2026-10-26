// Integer (INT8) inference of the QAT-converted intrusion-detection MLP on the M5Stack Basic (ESP32).
//
// Same serial protocol and timing as arduino_m5_ids_normal_exp6.ino (FP32): the host sends one CSV line
// per sample (61 features already rescaled on the host + label) and the board answers
// "PRED,<pred>,<label>,<latency_us>". The latency covers the quantization of the input, the 12 integer
// layers and the argmax, not the serial parsing.
//
// Arithmetic (see export_qat_int8.py, which reproduces it bit by bit in numpy before flashing):
//   input:  q = clamp(lrintf(x / s0) + z0, 0, 255)
//   layer:  acc = bias_i32[o] + sum_i (q_in[i] - z_in) * w[o][i]          (int32)
//           q_out = clamp(lrintf((float) acc * requant) + z_out, 0, 255)
//           hidden layers: q_out = max(q_out, z_out)                      (ReLU on a quantized tensor)
//   output: argmax over the uint8 codes of the last layer (first maximum wins, like torch.argmax)

#include <M5Unified.h>
#include <math.h>
#include <stdint.h>
#include "exp6_qat_int8_model.h"

#define MAX_NEURONS 128
#define MAX_INPUTS  64

static uint8_t qin[MAX_INPUTS];
static uint8_t buf_a[MAX_NEURONS];
static uint8_t buf_b[MAX_NEURONS];

static inline int clamp_u8(int v) {
  return v < 0 ? 0 : (v > 255 ? 255 : v);
}

void linear_forward_int8(
  const uint8_t* input,
  int in_dim,
  int in_zp,
  const int8_t* W,
  const int32_t* B,
  int out_dim,
  float requant,
  int out_zp,
  bool apply_relu,
  uint8_t* output
) {
  for (int o = 0; o < out_dim; ++o) {
    int32_t acc = B[o];
    const int8_t* row = W + o * in_dim;
    for (int i = 0; i < in_dim; ++i) {
      acc += ((int32_t)input[i] - in_zp) * (int32_t)row[i];
    }
    int q = clamp_u8((int)lrintf((float)acc * requant) + out_zp);
    if (apply_relu && q < out_zp) q = out_zp;
    output[o] = (uint8_t)q;
  }
}

int argmax_u8(const uint8_t* x, int n) {
  int idx = 0;
  uint8_t best = x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] > best) {
      best = x[i];
      idx = i;
    }
  }
  return idx;
}

bool read_sample_from_serial(float* features, int expected_dim, int& true_label) {
  if (!Serial.available()) return false;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return false;

  int start = 0;
  int col = 0;
  int feature_idx = 0;
  true_label = -1;

  while (start < line.length()) {
    int comma = line.indexOf(',', start);
    if (comma == -1) comma = line.length();

    String token = line.substring(start, comma);
    token.trim();

    if (col < expected_dim) {
      features[feature_idx++] = token.toFloat();
    } else {
      true_label = token.toInt();
    }

    col++;
    start = comma + 1;
  }

  return (feature_idx == expected_dim && true_label >= 0);
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);

  Serial.begin(115200);
  delay(1000);

  M5.Display.setTextSize(2);
  M5.Display.setCursor(20, 20);
  M5.Display.println("IDS INT8");

  Serial.println("READY");
}

void loop() {
  const int* sizes = exp6_qat_int8_layer_sizes;
  const int num_layers = exp6_qat_int8_num_layers;
  const int input_dim = sizes[0];
  const int output_dim = sizes[num_layers];

  static float input[MAX_INPUTS];
  int true_label = -1;

  if (!read_sample_from_serial(input, input_dim, true_label)) {
    delay(1);
    return;
  }

  unsigned long t0 = micros();

  for (int i = 0; i < input_dim; ++i) {
    qin[i] = (uint8_t)clamp_u8((int)lrintf(input[i] / exp6_qat_int8_input_scale) + exp6_qat_int8_input_zero_point);
  }

  const uint8_t* current_in = qin;
  uint8_t* current_out = buf_a;

  for (int layer = 0; layer < num_layers; ++layer) {
    linear_forward_int8(
      current_in,
      sizes[layer],
      exp6_qat_int8_in_zero_point[layer],
      exp6_qat_int8_weights[layer],
      exp6_qat_int8_biases[layer],
      sizes[layer + 1],
      exp6_qat_int8_requant[layer],
      exp6_qat_int8_out_zero_point[layer],
      layer < num_layers - 1,
      current_out
    );
    current_in = current_out;
    current_out = (current_out == buf_a) ? buf_b : buf_a;
  }

  int pred = argmax_u8(current_in, output_dim);

  unsigned long t1 = micros();
  unsigned long latency_us = t1 - t0;

  Serial.print("PRED,");
  Serial.print(pred);
  Serial.print(",");
  Serial.print(true_label);
  Serial.print(",");
  Serial.println(latency_us);
}
