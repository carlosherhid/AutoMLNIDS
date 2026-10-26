#include <M5Unified.h>
#include "exp6_normal_model.h"

#define MAX_NEURONS 128
#define MAX_INPUTS  64

float buf_a[MAX_NEURONS];
float buf_b[MAX_NEURONS];

float relu(float x) {
  return x > 0.0f ? x : 0.0f;
}

void linear_forward(
  const float* input,
  int in_dim,
  const float* W,
  const float* B,
  int out_dim,
  float* output,
  bool apply_relu
) {
  for (int o = 0; o < out_dim; ++o) {
    float sum = B[o];
    for (int i = 0; i < in_dim; ++i) {
      sum += input[i] * W[o * in_dim + i];
    }
    output[o] = apply_relu ? relu(sum) : sum;
  }
}

int argmax_vec(const float* x, int n) {
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
  M5.Display.println("IDS EXP6");

  Serial.println("READY");
}

void loop() {
  const int* sizes = exp6_normal_model_layer_sizes;
  const int num_layers = exp6_normal_model_num_layers;
  const int input_dim = sizes[0];
  const int output_dim = sizes[num_layers];

  static float input[MAX_INPUTS];
  int true_label = -1;

  if (!read_sample_from_serial(input, input_dim, true_label)) {
    delay(1);
    return;
  }

  unsigned long t0 = micros();

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
      default:
        Serial.println("ERR,LAYER_INDEX");
        return;
    }

    linear_forward(current_in, in_dim, W, B, out_dim, current_out, apply_relu);

    // swap after every layer, the last one included, so that current_in points to the output of the last layer
    // (with the swap restricted to the hidden layers, the logits read below were the activations of layer 10)
    if (current_out == buf_a) {
      current_in = buf_a;
      current_out = buf_b;
    } else {
      current_in = buf_b;
      current_out = buf_a;
    }
  }

  const float* logits = current_in;
  int pred = argmax_vec(logits, output_dim);

  unsigned long t1 = micros();
  unsigned long latency_us = t1 - t0;

  Serial.print("PRED,");
  Serial.print(pred);
  Serial.print(",");
  Serial.print(true_label);
  Serial.print(",");
  Serial.println(latency_us);
}