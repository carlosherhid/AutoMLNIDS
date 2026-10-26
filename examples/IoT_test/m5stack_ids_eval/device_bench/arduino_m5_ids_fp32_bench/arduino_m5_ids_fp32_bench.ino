// Benchmark build of the FP32 intrusion-detection MLP on the M5Stack Basic (ESP32).
// Same float arithmetic as arduino_m5_ids_normal_exp6.ino; what changes is the serial protocol and the measurements,
// defined in bench_common.h (see arduino_m5_ids_int8_bench.ino for the description).

#include "bench_common.h"
#include "exp6_normal_model.h"

#define MAX_NEURONS 128

static float buf_a[MAX_NEURONS];
static float buf_b[MAX_NEURONS];

static inline float relu(float x) {
  return x > 0.0f ? x : 0.0f;
}

static void linear_forward(const float* input, int in_dim, const float* W, const float* B, int out_dim, float* output,
                           bool apply_relu) {
  for (int o = 0; o < out_dim; ++o) {
    float sum = B[o];
    for (int i = 0; i < in_dim; ++i) {
      sum += input[i] * W[o * in_dim + i];
    }
    output[o] = apply_relu ? relu(sum) : sum;
  }
}

static int argmax_vec(const float* x, int n) {
  int idx = 0;
  float best = x[0];
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
  const int* sizes = exp6_normal_model_layer_sizes;
  const int num_layers = exp6_normal_model_num_layers;
  const int input_dim = sizes[0];
  const int output_dim = sizes[num_layers];

  rescale_features(input, input_dim);

  const float* current_in = input;
  float* current_out = buf_a;
  for (int layer = 0; layer < num_layers; ++layer) {
    int in_dim = sizes[layer];
    int out_dim = sizes[layer + 1];
    bool apply_relu = (layer < num_layers - 1);

    const float* W = nullptr;
    const float* B = nullptr;
    switch (layer) {
      case 0:  W = exp6_normal_model_w_0;  B = exp6_normal_model_b_0;  break;
      case 1:  W = exp6_normal_model_w_1;  B = exp6_normal_model_b_1;  break;
      case 2:  W = exp6_normal_model_w_2;  B = exp6_normal_model_b_2;  break;
      case 3:  W = exp6_normal_model_w_3;  B = exp6_normal_model_b_3;  break;
      case 4:  W = exp6_normal_model_w_4;  B = exp6_normal_model_b_4;  break;
      case 5:  W = exp6_normal_model_w_5;  B = exp6_normal_model_b_5;  break;
      case 6:  W = exp6_normal_model_w_6;  B = exp6_normal_model_b_6;  break;
      case 7:  W = exp6_normal_model_w_7;  B = exp6_normal_model_b_7;  break;
      case 8:  W = exp6_normal_model_w_8;  B = exp6_normal_model_b_8;  break;
      case 9:  W = exp6_normal_model_w_9;  B = exp6_normal_model_b_9;  break;
      case 10: W = exp6_normal_model_w_10; B = exp6_normal_model_b_10; break;
      case 11: W = exp6_normal_model_w_11; B = exp6_normal_model_b_11; break;
      default: return -1;
    }

    linear_forward(current_in, in_dim, W, B, out_dim, current_out, apply_relu);

    if (layer < num_layers - 1) {
      current_in = current_out;
      current_out = (current_out == buf_a) ? buf_b : buf_a;
    } else {
      current_in = current_out;
    }
  }
  return argmax_vec(current_in, output_dim);
}

void setup() {
  bench_setup("IDS FP32 bench");
}

void loop() {
  bench_loop(infer, exp6_normal_model_layer_sizes[0]);
}
