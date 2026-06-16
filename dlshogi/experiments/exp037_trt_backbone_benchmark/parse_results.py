"""Parse trtexec logs into a latency/throughput comparison table."""
import re
import sys
from pathlib import Path
from collections import defaultdict

# trtexec reports lines like:
#   GPU Compute Time: min = ..., max = ..., mean = 1.234 ms, median = ..., percentile(99%) = ...
#   Throughput: 12345.6 qps
LAT_RE = re.compile(r"GPU Compute Time:.*?mean = ([\d.]+) ms.*?median = ([\d.]+) ms", re.IGNORECASE)
THR_RE = re.compile(r"Throughput:\s*([\d.]+)\s*qps", re.IGNORECASE)
NAME_RE = re.compile(r"(?P<net>.+)_b(?P<bs>\d+)\.log$")

NET_ORDER = ["inceptionnext", "resnet32x384", "senet32x384", "hybrid36p5", "hybrid37p4", "hybrid_exp038"]


def parse_log(path):
    text = path.read_text()
    lat = LAT_RE.search(text)
    thr = THR_RE.search(text)
    mean = float(lat.group(1)) if lat else None
    median = float(lat.group(2)) if lat else None
    qps = float(thr.group(1)) if thr else None
    return mean, median, qps


def main(resdir):
    resdir = Path(resdir)
    # data[bs][net] = (mean_ms, median_ms, qps)
    data = defaultdict(dict)
    for log in sorted(resdir.glob("*_b*.log")):
        m = NAME_RE.search(log.name)
        if not m:
            continue
        net, bs = m.group("net"), int(m.group("bs"))
        data[bs][net] = parse_log(log)

    print()
    print(f"{'batch':>6} | {'network':<16} | {'mean ms':>9} | {'median ms':>10} | {'qps':>12} | {'pos/s':>12} | {'vs INX':>7}")
    print("-" * 90)
    for bs in sorted(data):
        inx = data[bs].get("inceptionnext")
        inx_thr = (bs * 1000.0 / inx[0]) if inx and inx[0] else None
        for net in NET_ORDER:
            if net not in data[bs]:
                continue
            mean, median, qps = data[bs][net]
            if mean is None:
                print(f"{bs:>6} | {net:<16} | {'FAIL':>9} | {'-':>10} | {'-':>12} | {'-':>12} | {'-':>7}")
                continue
            pos_s = bs * 1000.0 / mean  # positions per second
            speedup = (pos_s / inx_thr) if inx_thr else float("nan")
            qps_s = f"{qps:.1f}" if qps is not None else "-"
            print(f"{bs:>6} | {net:<16} | {mean:>9.4f} | {median:>10.4f} | {qps_s:>12} | {pos_s:>12.1f} | {speedup:>6.2f}x")
        print("-" * 90)
    print("\nNotes:")
    print("- mean/median ms = GPU compute time per inference call (lower is better).")
    print("- pos/s = batch / mean_latency = positions evaluated per second (higher is better).")
    print("- vs INX = pos/s relative to InceptionNeXt at the same batch (>1 = faster than InceptionNeXt).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
