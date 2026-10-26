// Benchmark build of the INT8 (QAT-converted) intrusion-detection MLP on the M5Stack Basic (ESP32).
// Same integer arithmetic as arduino_m5_ids_int8.ino (see export_qat_int8.py); what changes is the serial protocol and
// the measurements, defined in bench_common.h: the near-zero rescaling is done on the board, the latency is reported
// per sample, heap and stack are reported, and a BENCH command runs a self-paced stream to measure CPU load and
// throughput without serial traffic.
//
//   input:  q = clamp(lrintf(x / s0) + z0, 0, 255)
//   layer:  acc = bias_i32[o] + sum_i (q_in[i] - z_in) * w[o][i]          (int32)
//           q_out = clamp(lrintf((float) acc * requant) + z_out, 0, 255)
//           hidden layers: q_out = max(q_out, z_out)
//   output: argmax over the uint8 codes of the last layer

#include "bench_common.h"
#include "exp6_qat_int8_model.h"

#define MAX_NEURONS 128

static uint8_t qin[MAX_INPUTS];
static uint8_t buf_a[MAX_NEURONS];
static uint8_t buf_b[MAX_NEURONS];

static inline int clamp_u8(int v) {
  return v < 0 ? 0 : (v > 255 ? 255 : v);
}

static void linear_forward_int8(const uint8_t* input, int in_dim, int in_zp, const int8_t* W, const int32_t* B,
                                int out_dim, float requant, int out_zp, bool apply_relu, uint8_t* output) {
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

static int argmax_u8(const uint8_t* x, int n) {
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

// raw Min-Max features in, predicted class out; the features are rescaled in place
static int infer(float* input) {
  const int* sizes = exp6_qat_int8_layer_sizes;
  const int num_layers = exp6_qat_int8_num_layers;
  const int input_dim = sizes[0];
  const int output_dim = sizes[num_layers];

  rescale_features(input, input_dim);

  for (int i = 0; i < input_dim; ++i) {
    qin[i] = (uint8_t)clamp_u8((int)lrintf(input[i] / exp6_qat_int8_input_scale) + exp6_qat_int8_input_zero_point);
  }

  const uint8_t* current_in = qin;
  uint8_t* current_out = buf_a;
  for (int layer = 0; layer < num_layers; ++layer) {
    linear_forward_int8(current_in, sizes[layer], exp6_qat_int8_in_zero_point[layer], exp6_qat_int8_weights[layer],
                        exp6_qat_int8_biases[layer], sizes[layer + 1], exp6_qat_int8_requant[layer],
                        exp6_qat_int8_out_zero_point[layer], layer < num_layers - 1, current_out);
    current_in = current_out;
    current_out = (current_out == buf_a) ? buf_b : buf_a;
  }
  return argmax_u8(current_in, output_dim);
}

void setup() {
  bench_setup("IDS INT8 bench");
}

void loop() {
  bench_loop(infer, exp6_qat_int8_layer_sizes[0]);
}
