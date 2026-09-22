from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass
class HPatchesPair:
    seq_name: str
    seq_type: str  
    ref_path: Path
    tgt_path: Path
    tgt_index: int
    H: np.ndarray  


def list_seq(root):
    root = Path(root)
    seqs = sorted(p for p in root.iterdir() if p.is_dir() and (p.name.startswith("i_") or p.name.startswith("v_")))
    if not seqs:
        raise FileNotFoundError(f"No HPatches sequences found under {root}")
    return seqs


def load_pairs(root, max_sequences=None, seed_order=True):
    
    seqs = list_seq(root)
    if seed_order:
        seqs = sorted(seqs, key=lambda p: p.name)
    if max_sequences is not None:
        seqs = seqs[:max_sequences]

    pairs = []
    for seq_dir in seqs:
        seq_type = seq_dir.name[0]
        ref_path = seq_dir / "1.ppm"
        if not ref_path.exists():
            continue
        for i in range(2, 7):
            tgt_path = seq_dir / f"{i}.ppm"
            h_path = seq_dir / f"H_1_{i}"
            if not tgt_path.exists() or not h_path.exists():
                continue
            H = np.loadtxt(h_path).astype(np.float64)
            pairs.append(
                HPatchesPair(
                    seq_name=seq_dir.name,
                    seq_type=seq_type,
                    ref_path=ref_path,
                    tgt_path=tgt_path,
                    tgt_index=i,
                    H=H,
                )
            )
    return pairs


def read_float(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray, bgr
