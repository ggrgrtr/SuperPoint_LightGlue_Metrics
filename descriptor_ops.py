import numpy as np
import torch


class PCA:
    def __init__(self, k):
        self.k = k
        self.mean_ = None
        self.components_ = None # матрица для 128,64, 32 компонент

    def fit(self, descriptors):
        X = np.asarray(descriptors, dtype=np.float64)
        self.mean_ = X.mean(axis=0, keepdims=True)
        Xc = X - self.mean_ # центровка дескр
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[: self.k] 
        return self

    def reconstruct(self, desc_t):
        device, dtype = desc_t.device, desc_t.dtype
        X = desc_t.detach().cpu().numpy().astype(np.float64)
        shape = X.shape
        flat = X.reshape(-1, shape[-1])
        Xc = flat - self.mean_
        proj = Xc @ self.components_.T 
        recon = proj @ self.components_ + self.mean_ 
        recon = recon.reshape(shape)
        return torch.from_numpy(recon).to(device=device, dtype=dtype)


def quant_dequant(desc, bits):
    if bits >= 32:
        return desc
    levels = 2 ** bits
    dmin = desc.min(dim=-1, keepdim=True).values
    dmax = desc.max(dim=-1, keepdim=True).values
    scale = (dmax - dmin).clamp(min=1e-12) / (levels - 1) 
    q = torch.round((desc - dmin) / scale).clamp(0, levels - 1)
    return q * scale + dmin


def renormalize(t):
    return torch.nn.functional.normalize(t, p=2, dim=-1)
