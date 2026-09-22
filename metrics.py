import cv2
import numpy as np


def warp_points(H, pts):
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    proj = (H @ pts_h.T).T
    return proj[:, :2] / proj[:, 2:3]


def compute_mma(mkpts0, mkpts1, H_gt, thresholds=(1, 3, 5, 10)):
    n = len(mkpts0)
    if n == 0:
        return {t: 0.0 for t in thresholds}, 0
    proj = warp_points(H_gt, mkpts0)
    dists = np.linalg.norm(proj - mkpts1, axis=1)
    return {t: float((dists < t).mean()) for t in thresholds}, n


def corner_error(mkpts0, mkpts1, H_gt, image_shape, ransac_thresh=3.0):
    h, w = image_shape
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    gt_warp = warp_points(H_gt, corners)

    if len(mkpts0) < 4:
        return float("inf")
    H_est, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransac_thresh)
    if H_est is None:
        return float("inf")
    est_warp = warp_points(H_est, corners)
    return float(np.linalg.norm(est_warp - gt_warp, axis=1).mean())


def error_auc(errors, thresholds):
    errors = np.array(sorted(errors), dtype=np.float64)
    n = len(errors)
    result = {}
    if n == 0:
        return {t: 0.0 for t in thresholds}
    errors_ext = np.concatenate([[0.0], errors])
    recall = np.linspace(0, 1, n + 1)
    for t in thresholds:
        last = np.searchsorted(errors_ext, t)
        if last == 0:
            result[t] = 0.0
            continue
        x = np.concatenate([errors_ext[:last], [t]])
        y = np.concatenate([recall[:last], [recall[last - 1]]])
        result[t] = float(np.trapezoid(y, x) / t)
    return result
