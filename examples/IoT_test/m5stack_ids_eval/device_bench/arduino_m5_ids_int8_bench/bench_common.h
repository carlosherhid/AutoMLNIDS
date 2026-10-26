// Shared code of the benchmark builds (INT8 and FP32) of the intrusion-detection MLP on the M5Stack Basic.
//
// Serial protocol (115200 -> SERIAL_BAUD, one text line per message, host -> board):
//   <61 raw Min-Max features>,<label>     classify one sample; the board applies the near-zero rescaling itself
//                                         (RESCALE_ON_DEVICE = 1) and answers
//                                         PRED,<pred>,<label>,<latency_us>,<rescale_us>
//   BENCH,<rate_hz>,<seconds>             self-paced run on the 100 samples stored in flash, one inference every
//                                         1/rate_hz s, no serial traffic while it runs; answers
//                                         BENCH,<rate_hz>,<seconds>,<n>,<misses>,<busy_us>,<elapsed_us>,<mean_us>,<max_us>,
//                                               <rescale_mean_us>,<free_heap>,<min_free_heap>,<stack_free_bytes>
//   STAT                                  answers STAT,<free_heap>,<min_free_heap>,<heap_size>,<stack_free_bytes>,<n_samples>
// The board also sends STAT every STAT_EVERY samples. "READY" is sent after boot.
//
// latency_us covers the rescaling (if enabled), the input quantization (INT8 build), the 12 layers and the argmax;
// it does not cover the serial parsing. rescale_us is the part of latency_us spent in the rescaling.
#pragma once
#include <M5Unified.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "bench_samples.h"

#ifndef SERIAL_BAUD
#define SERIAL_BAUD 921600
#endif
#ifndef RESCALE_ON_DEVICE
#define RESCALE_ON_DEVICE 1
#endif
#ifndef STAT_EVERY
#define STAT_EVERY 5000
#endif
#define MAX_INPUTS 64

typedef int (*infer_fn_t)(float*);

static uint32_t g_rescale_us = 0;  // time spent rescaling the last sample
static uint32_t g_samples = 0;     // samples classified through the serial port

// Equation 2 of the paper in float32: x' = 1e6 * x if |x| < 1e-6, x otherwise.
static inline float rescale_value(float x) {
  return (fabsf(x) < 1e-6f) ? x * 1e6f : x;
}

// Rescales the raw Min-Max features in place and records how long it took.
static inline void rescale_features(float* x, int n) {
#if RESCALE_ON_DEVICE
  uint32_t t0 = micros();
  for (int i = 0; i < n; ++i) x[i] = rescale_value(x[i]);
  g_rescale_us = micros() - t0;
#else
  (void)x; (void)n;
  g_rescale_us = 0;
#endif
}

static void print_stat() {
  Serial.print("STAT,");
  Serial.print((unsigned long)ESP.getFreeHeap());
  Serial.print(",");
  Serial.print((unsigned long)ESP.getMinFreeHeap());
  Serial.print(",");
  Serial.print((unsigned long)ESP.getHeapSize());
  Serial.print(",");
  Serial.print((unsigned long)uxTaskGetStackHighWaterMark(NULL));  // free stack of the Arduino loop task at its minimum, bytes
  Serial.print(",");
  Serial.println((unsigned long)g_samples);
}

// "f0,f1,...,f60,label" -> features (parsed as double and cast to float, like the host does) and label.
static bool parse_sample(const String& line, float* f, int dim, int& label) {
  const char* p = line.c_str();
  char* end = nullptr;
  for (int i = 0; i < dim; ++i) {
    f[i] = (float)strtod(p, &end);
    if (end == p || *end != ',') return false;
    p = end + 1;
  }
  label = (int)strtol(p, &end, 10);
  return end != p && label >= 0;
}

static void run_bench(infer_fn_t infer, int rate_hz, int seconds) {
  static float work[MAX_INPUTS];
  if (rate_hz < 1) rate_hz = 1;
  const uint32_t period = 1000000UL / (uint32_t)rate_hz;
  const uint32_t t_start = micros();
  const uint32_t t_end = t_start + (uint32_t)seconds * 1000000UL;
  uint32_t next = t_start;
  uint32_t n = 0, misses = 0, max_lat = 0;
  uint64_t busy = 0, rescale_sum = 0;
  int idx = 0;
  while ((int32_t)(micros() - t_end) < 0) {
    while ((int32_t)(micros() - next) < 0) { /* the CPU waits for the next sample to "arrive" */ }
    memcpy(work, bench_samples[idx], BENCH_DIM * sizeof(float));
    uint32_t t0 = micros();
    infer(work);
    uint32_t lat = micros() - t0;
    busy += lat;
    rescale_sum += g_rescale_us;
    if (lat > max_lat) max_lat = lat;
    n++;
    idx = (idx + 1) % BENCH_N;
    next += period;
    if ((int32_t)(micros() - next) > 0) misses++;  // finished after the next arrival: the backlog grows
  }
  const uint32_t elapsed = micros() - t_start;
  Serial.print("BENCH,");
  Serial.print(rate_hz); Serial.print(",");
  Serial.print(seconds); Serial.print(",");
  Serial.print((unsigned long)n); Serial.print(",");
  Serial.print((unsigned long)misses); Serial.print(",");
  Serial.print((unsigned long)busy); Serial.print(",");
  Serial.print((unsigned long)elapsed); Serial.print(",");
  Serial.print(n ? (double)busy / n : 0.0, 2); Serial.print(",");
  Serial.print((unsigned long)max_lat); Serial.print(",");
  Serial.print(n ? (double)rescale_sum / n : 0.0, 2); Serial.print(",");
  Serial.print((unsigned long)ESP.getFreeHeap()); Serial.print(",");
  Serial.print((unsigned long)ESP.getMinFreeHeap()); Serial.print(",");
  Serial.println((unsigned long)uxTaskGetStackHighWaterMark(NULL));
}

static void bench_setup(const char* title) {
  auto cfg = M5.config();
  cfg.serial_baudrate = 0;       // M5.begin() must not open the serial port: its RX buffer is sized below
  M5.begin(cfg);
  Serial.setRxBufferSize(4096);  // one sample line is about 600-800 bytes; has to be called before begin()
  Serial.begin(SERIAL_BAUD);
  delay(1000);
  M5.Display.setTextSize(2);
  M5.Display.setCursor(20, 20);
  M5.Display.println(title);
  Serial.println("READY");
}

// One iteration of loop(): handles BENCH / STAT commands and samples.
static void bench_loop(infer_fn_t infer, int input_dim) {
  if (!Serial.available()) {
    delayMicroseconds(100);
    return;
  }
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (line.startsWith("BENCH,")) {
    int c = line.indexOf(',', 6);
    if (c < 0) { Serial.println("ERR,BENCH_ARGS"); return; }
    run_bench(infer, line.substring(6, c).toInt(), line.substring(c + 1).toInt());
    return;
  }
  if (line.startsWith("STAT")) {
    print_stat();
    return;
  }

  static float input[MAX_INPUTS];
  int label = -1;
  if (!parse_sample(line, input, input_dim, label)) {
    Serial.println("ERR,PARSE");
    return;
  }
  uint32_t t0 = micros();
  int pred = infer(input);
  uint32_t latency_us = micros() - t0;
  g_samples++;

  Serial.print("PRED,");
  Serial.print(pred);
  Serial.print(",");
  Serial.print(label);
  Serial.print(",");
  Serial.print((unsigned long)latency_us);
  Serial.print(",");
  Serial.println((unsigned long)g_rescale_us);
  if (g_samples % STAT_EVERY == 0) print_stat();
}
