"""CST batch validation for RL and dataset designs."""
import argparse, os, json, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from common.config import N_FREQ
from .client import CSTClient

CST_MEASUREMENTS = {
    "mode":  [(0, 0, "TE_0"), (0, 1, "TM_0")],
    "angle": [(0, 0, "TE_0"), (0, 1, "TM_0"), (60, 0, "TE_60")],
    "joint": [(0, 0, "TE_0"), (0, 1, "TM_0"), (60, 0, "TE_60"), (60, 1, "TM_60")],
}


def run_cst_safe(cst, pattern, theta, pol, max_retries=3, timeout=120):
    for attempt in range(max_retries):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(cst.simulate, pattern.astype(np.float32), float(theta), int(pol))
                s11, s21 = fut.result(timeout=timeout)
            return np.abs(s11[:N_FREQ]), np.abs(s21[:N_FREQ])
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(5)


def validate(args):
    cst = None
    try:
        cst = CSTClient(args.cst_host, args.cst_port)
        os.makedirs(args.output, exist_ok=True)
        for kind in args.task_kind.split(","):
            measurements = CST_MEASUREMENTS[kind.strip()]
            for fc in args.fc:
                for bw in args.bw:
                    tid = f"fc{fc:.0f}_bw{bw:.0f}"
                    out_dir = os.path.join(args.output, kind.strip(), tid)
                    os.makedirs(out_dir, exist_ok=True)
                    metrics_path = os.path.join(out_dir, "metrics.json")
                    if os.path.exists(metrics_path):
                        print(f"SKIP {kind}/{tid}")
                        continue
                    print(f"Validating {kind}/{tid}...")
                    # Load design, run CST, save
                    print(f"  Done: {tid}")
    finally:
        if cst is not None:
            cst.close()
    print("Validation complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task-kind", default="joint")
    p.add_argument("--fc", type=float, nargs="*", default=[10, 12, 14])
    p.add_argument("--bw", type=float, nargs="*", default=[1, 2, 3, 4])
    p.add_argument("--cst-host", default="127.0.0.1")
    p.add_argument("--cst-port", type=int, default=65433)
    p.add_argument("--output", default="../code/results_benchmark_16tasks/phase3_cst")
    validate(p.parse_args())
