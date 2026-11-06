"""Brush PLY (3DGS, any SH degree) to the 32-byte .splat format the browser explorer loads.

    python -m drivingbench.sim.explorer.convert SPLAT.ply OUT.splat [--max-splats 1500000]
                                                [--radius 300]

Keeps the base colour (SH0) and drops view-dependent terms; prunes anything beyond `radius`
metres of the lot, then keeps the `max-splats` most opaque (fine detail survives; selecting by
size would keep the large background blobs instead). Coordinates stay in the course frame.
"""

import argparse
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814
CENTRE = np.array([60.0, 10.0])  # the lot, in the course frame


def read_ply(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        lines = header.splitlines()
        props = [line.split()[-1].decode() for line in lines if line.startswith(b"property")]
        count = int(next(line for line in lines if line.startswith(b"element vertex")).split()[-1])
        data = np.frombuffer(f.read(count * 4 * len(props)), np.float32).reshape(count, len(props))
    return {name: data[:, i] for i, name in enumerate(props)}


def convert(ply: Path, out: Path, max_splats: int, radius: float) -> int:
    c = read_ply(ply)
    opacity = 1 / (1 + np.exp(-c["opacity"]))
    xyz = np.stack([c["x"], c["y"], c["z"]], axis=1)
    near = np.flatnonzero(np.linalg.norm(xyz[:, :2] - CENTRE, axis=1) <= radius)
    keep = np.zeros(len(opacity), bool)
    keep[near[np.argsort(-opacity[near], kind="stable")[:max_splats]]] = True
    scale = np.exp(np.stack([c[f"scale_{i}"] for i in range(3)], axis=1))
    order = np.argsort(-(scale.prod(axis=1) * opacity)[keep])  # important splats first
    idx = np.flatnonzero(keep)[order]
    rgb = 0.5 + SH_C0 * np.stack([c[f"f_dc_{i}"] for i in range(3)], axis=1)
    rot = np.stack([c[f"rot_{i}"] for i in range(4)], axis=1)  # w, x, y, z
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    record = np.zeros(len(idx), dtype=[("xyz", "<f4", 3), ("scale", "<f4", 3),
                                        ("rgba", "u1", 4), ("rot", "u1", 4)])  # fmt: skip
    record["xyz"], record["scale"] = xyz[idx], scale[idx]
    record["rgba"][:, :3] = np.clip(rgb[idx] * 255, 0, 255)
    record["rgba"][:, 3] = np.clip(opacity[idx] * 255, 0, 255)
    record["rot"] = np.clip(rot[idx] * 128 + 128, 0, 255)
    out.write_bytes(record.tobytes())
    return len(idx)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("ply", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--max-splats", type=int, default=1_500_000)
    a.add_argument("--radius", type=float, default=300.0)
    args = a.parse_args()
    n = convert(args.ply, args.out, args.max_splats, args.radius)
    print(f"{n} splats, {args.out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
