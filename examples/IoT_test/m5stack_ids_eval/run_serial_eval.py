import argparse
import csv
import serial
import time

def wait_for_ready(ser, max_lines=50):
    print("Esperando READY...")
    for _ in range(max_lines):
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("DEVICE:", line)
        if "READY" in line:
            return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    print(f"Abriendo puerto {args.port} a {args.baud} baud...")
    ser = serial.Serial(args.port, args.baud, timeout=2)

    # Espera inicial
    time.sleep(3)

    got_ready = wait_for_ready(ser)
    if not got_ready:
        print("READY not received automatically.")
        print("Press RESET on the M5Stack now, if needed...")
        got_ready = wait_for_ready(ser, max_lines=80)

    if not got_ready:
        print("READY not received, continuing anyway.")

    with open(args.input_csv, "r", encoding="utf-8") as fin, \
         open(args.output_csv, "w", newline="", encoding="utf-8") as fout:

        writer = csv.writer(fout)
        writer.writerow(["pred", "true", "latency_us"])

        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            ser.write((line + "\n").encode())
            ser.flush()

            got_pred = False
            for _ in range(200):
                resp = ser.readline().decode(errors="ignore").strip()
                if not resp:
                    continue

                print("DEVICE:", resp)

                if resp.startswith("PRED,"):
                    _, pred, true_label, latency_us = resp.split(",")
                    writer.writerow([int(pred), int(true_label), int(latency_us)])
                    print(f"[OK] sample={idx} pred={pred} true={true_label} latency_us={latency_us}")
                    got_pred = True
                    break

            if not got_pred:
                print(f"[WARN] PRED not received for sample {idx}")
                writer.writerow([-1, -1, -1])

    ser.close()
    print("Evaluation finished.")

if __name__ == "__main__":
    main()
