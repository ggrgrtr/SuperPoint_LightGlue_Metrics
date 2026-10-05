import csv
from dataclasses import dataclass
import sys
from pathlib import Path
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from descriptor_ops import PCA, quant_dequant, renormalize
from hpatches_dataset import list_seq, load_pairs, read_float
from metrics import compute_mma, corner_error, error_auc


MMA_THRESHOLDS = (1, 3, 5, 10)
AUC_THRESHOLDS = (3, 5, 10)
PCA_DIMS = (128, 64, 32)
QUANT_BITS = (4, 2)
VARIANT_FIELD = "вариант"
AVG_MATCHES_FIELD = "средн. совпадений"
BASELINE_NAME = "база_256d_fp32"


@dataclass(frozen=True)
class conf:
    data_dir: str = "data/hpatches-sequences-release"
    max_sequences: int | None = None #20
    calib_stride: int = 6
    pca_dims: tuple[int, ...] = PCA_DIMS
    quant_bits: tuple[int, ...] = QUANT_BITS
    max_keypoints: int = 1024
    resize: int = 1024
    output: str | None = "ablation_results.csv"
    txt_: str | None = "a.txt"


def build_variants(pca_reducers, quant_bits):
    variants = [{"name": BASELINE_NAME, "transform": None}]
    for k, reducer in pca_reducers.items():
        variants.append({"name": f"pca_{k}_параметров", "transform": ("pca", reducer)})
    for bits in quant_bits:
        variants.append({"name": f"квантование_{bits}_бит", "transform": ("quant", bits)})
    return variants


def apply_transform(desc, transform):
    if transform is None:
        return desc
    kind, arg = transform
    if kind == "pca":
        out = arg.reconstruct(desc)
    elif kind == "quant":
        out = quant_dequant(desc, bits=arg)
    else:
        raise ValueError(kind)
    return renormalize(out)


def to_tensor(path):
    gray, bgr = read_float(path)
    return torch.from_numpy(gray)[None], bgr.shape[:2]  # (1,H,W), (H,W)


@torch.no_grad()
def extract_features(extractor, path, resize, cache):
    key = str(path)
    if key in cache:
        return cache[key]
    img_t, hw = to_tensor(path)
    feats = extractor.extract(img_t, resize=resize)
    cache[key] = (feats, hw)
    return feats, hw


@torch.no_grad() # калибровка, обучение PCA на дескрипторах опорных изображений
def fit_pca(extractor, calib_seqs, dims, resize):
    print(f"! обучение PCA на дескрипторах опорных изображений: {len(calib_seqs)} шт")
    pool = []
    for seq in calib_seqs:
        ref = seq / "1.ppm"
        if not ref.exists():
            continue
        img_t, _ = to_tensor(ref)
        feats = extractor.extract(img_t, resize=resize)
        pool.append(feats["descriptors"][0].cpu().numpy())
    all_desc = np.concatenate(pool, axis=0)
    print(f"! собрано {all_desc.shape[0]} дескрипторов для обучения PCA")
    a = {}
    for k in dims:
        a[k] = PCA(k).fit(all_desc)
    return a


def string_metric(rows, header):
    baseline = rows[0]
    metric_keys = [k for k in header if k != VARIANT_FIELD]
    s=[]
    s.append(f"\n=== Падение метрик относительно {BASELINE_NAME} ===\n")
    drop_header = ["вариант", "метрика", "база", "значение", "падение", "падение_%\n"]
    s.append("  |  ".join(f"{h:>16}" for h in drop_header))
    for row in rows[1:]:
        for key in metric_keys:
            baseline_value = baseline[key]
            value = row[key]
            abs_drop = baseline_value - value
            rel_drop = 100.0 * abs_drop / baseline_value if baseline_value != 0 else 0.0
            s.append(
                f"{row[VARIANT_FIELD]:>16} | "
                f"{key:>16} | "
                f"{baseline_value:>16.3f} | "
                f"{value:>16.3f} | "
                f"{abs_drop:>16.3f} | "
                f"{rel_drop:>15.1f}%\n"
            ) 
    return s



@torch.no_grad()
def run(config):
    device = "cpu"
    extractor = SuperPoint(max_num_keypoints=config.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    print(
        "[настройки]\n"
        f"папка_данных = {config.data_dir}\nмаксимум_последовательностей = {config.max_sequences}\n"
        f"шаг_калибровки = {config.calib_stride}\nразмерности_pca = {config.pca_dims}\n"
        f"биты_квантования = {config.quant_bits}\nмаксимум_ключевых_точек = {config.max_keypoints}\n"
        f"resize = {config.resize}\nфайл_результатов = {config.output}\n"
    )


    all_seqs= list_seq(config.data_dir)
    calib_seqs= all_seqs[:: config.calib_stride]
    calib_names= {s.name for s in calib_seqs}
    remaining= [s for s in all_seqs if s.name not in calib_names]

    pairs = load_pairs(config.data_dir)
    pairs = [p for p in pairs if p.seq_name not in calib_names]
    if config.max_sequences is not None:
        keep_names={s.name for s in remaining[: config.max_sequences//2]}
        keep_names={s.name for s in remaining[-config.max_sequences//2:]}
        pairs = [p for p in pairs if p.seq_name in keep_names]

    print(f"! всего последовательностей: {len(all_seqs)} | калибровка PCA: {len(calib_seqs)}, "
          f"оценочных пар: {len(pairs)} из {len(set(p.seq_name for p in pairs))} последовательн.")

    pca_reducers = fit_pca(
        extractor,
        calib_seqs,
        config.pca_dims,
        config.resize,
    )
    variants = build_variants(pca_reducers, config.quant_bits)
    print(f"! варианты: "+", ".join(v["name"] for v in variants))

    
    stats = {
        v["name"]:{
            "mma":{threshold: [] for threshold in MMA_THRESHOLDS},
            "n_matches":[],
            "corner_err":[],
        }
        for v in variants
    }

    feat_cache = {}
    for i, pair in enumerate(pairs):
        # получение данных о ключ. точках изобржений пары
        feats0, hw0= extract_features(extractor, pair.ref_path, config.resize, feat_cache)
        feats1, hw1= extract_features(extractor, pair.tgt_path, config.resize, feat_cache)

        for v in variants:
            f0 = dict(feats0)
            f1 = dict(feats1)
            # замена дескр
            f0["descriptors"] = apply_transform(feats0["descriptors"], v["transform"])
            f1["descriptors"] = apply_transform(feats1["descriptors"], v["transform"])

            out = matcher({"image0": f0, "image1": f1})
            matches = out["matches"][0].cpu().numpy()
            kpts0 = feats0["keypoints"][0].cpu().numpy()
            kpts1 = feats1["keypoints"][0].cpu().numpy()
            mkpts0 = kpts0[matches[:, 0]]
            mkpts1 = kpts1[matches[:, 1]]

            mma_scores, n = compute_mma(mkpts0, mkpts1, pair.H, MMA_THRESHOLDS)
            for threshold in MMA_THRESHOLDS:
                stats[v["name"]]["mma"][threshold].append(mma_scores[threshold])
            stats[v["name"]]["n_matches"].append(n)

            cerr = corner_error(mkpts0, mkpts1, pair.H, hw0)
            stats[v["name"]]["corner_err"].append(cerr)

        if (i + 1) % 10 == 0 or i == len(pairs) - 1:            
            print(f"{i+1}/{len(pairs)}", flush=True)

    
    rows = []
    for v in variants:
        name = v["name"]
        s = stats[name]
        row = {VARIANT_FIELD: name, AVG_MATCHES_FIELD: float(np.mean(s["n_matches"]))}
        for threshold in MMA_THRESHOLDS:
            row[f"mma@{threshold}px"] = float(np.mean(s["mma"][threshold]))
        auc_scores = error_auc(s["corner_err"], AUC_THRESHOLDS)
        for threshold in AUC_THRESHOLDS:
            row[f"hauc@{threshold}px"] = auc_scores[threshold]
        rows.append(row)

    baseline = rows[0]
    print("\n=== Результат. макроусреднение по парам изображений ===")
    header = (
        [VARIANT_FIELD, AVG_MATCHES_FIELD]
        + [f"mma@{threshold}px" for threshold in MMA_THRESHOLDS]
        + [f"hauc@{threshold}px" for threshold in AUC_THRESHOLDS]
    )
    print(" | ".join(f"{h:>16}" for h in header))
    for row in rows:
        print(" | ".join(f"{row[h]:>16.3f}" if h != VARIANT_FIELD else f"{row[h]:>16}" for h in header))

    print(f"\n=== изменение относительно {BASELINE_NAME} % ===")
    metric_keys = [k for k in header if k != VARIANT_FIELD]
    print(" | ".join(f"{h:>16}" for h in header))
    
    for row in rows:
        cells = [f"{row[VARIANT_FIELD]:>16}"]
        for k in metric_keys:
            baseline_value = baseline[k]

            deff = 100.0 * (row[k] - baseline_value) / baseline_value if baseline_value != 0 else 0.0
            cells.append(f"{deff:>+15.1f}%")

        print(" | ".join(cells))


    for s in string_metric(rows, header):
        print(s,end='')

    if config.output:
        out_path = Path(config.output)
        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[сохранено] {out_path}")

    if config.txt_:
        out_path = Path(config.txt_)
        with out_path.open("w", encoding="utf-8") as f:
            for line in string_metric(rows, header):
                f.write(line)
        print(f"\n[сохранено] {out_path}")


    bl=rows[0]
    mx={'max':0}
    mn={'min':0}
    for i in rows[1:]:
        for j in header:
            if j==VARIANT_FIELD:
                continue
            v=i[j]
            v_bl=bl[j]
            abs_drop = v_bl - v
            rel_drop = 100.0 * abs_drop/v_bl if v_bl != 0 else 0.0

            if (mx[list(mx)[-1]]<=rel_drop):
                mx[i[VARIANT_FIELD]]=rel_drop
            if len(mx)>3:
                mx.pop(next(iter(mx)))

            if (mn[list(mn)[-1]]>=rel_drop):
                mn[i[VARIANT_FIELD]]=rel_drop
            if len(mn)>3:
                mn.pop(next(iter(mn)))

    print("\n\n")
    print("best scores:")
    print(*mn)
    print(*[mn[i] for i in mn])
    print('worst scores:')
    print(*mx)
    print(*[mx[i] for i in mx])

    return rows


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    run(conf())
