"""Polar FFT verifier: does a cluster's source region look like its copy?

For each RANSAC cluster, each side's keypoint bounding box is cut from the
image and resampled to PATCH x PATCH. The magnitude of its Fourier transform
does not change when the content shifts inside the window; resampled on a
log-polar grid, a rotation of the content becomes a shift along the angle axis
and a scale change a shift along the log-radius axis. Every radius of that grid
is normalised across angles (zero mean, unit spread), which removes the
fall-off with frequency that all natural images share and would otherwise make
any two regions look alike. The two spectra are compared by normalised
correlation, maximised over those shifts: 1 for identical content.

A cluster passes when its similarity is at least min_similarity. Clusters are
checked in RANSAC order (most inliers first), so the answer for the image is
the first passing cluster; any other passing cluster within tie_margin of its
similarity is kept as suspicious too.
"""
import numpy as np, cv2 as cv

PATCH = 64              # both sides are resampled to PATCH x PATCH before the FFT
MIN_SIDE = 16           # smallest box side cut from the image, px
N_RADIUS, N_ANGLE = 32, 64
MAX_RADIUS_SHIFT = 3    # log-radius rows searched either way (scale left after resampling)
MIN_SIMILARITY = 0.9    # provisional (local samples); calibrated by eval_fft.py
TIE_MARGIN = 0.02

_HANN = cv.createHanningWindow((PATCH, PATCH), cv.CV_32F)


def side_patch(gray, region):
  """A region's bounding box (each side at least MIN_SIDE, centred on the box)
  resampled to PATCH x PATCH; pixels past the image border are replicated.
  The box, not a square around it: a square around a tall or wide copy takes
  in surroundings that differ between the two sides."""
  x, y, w, h = region["bbox"]
  win = cv.getRectSubPix(gray, (max(w, MIN_SIDE), max(h, MIN_SIDE)), (x + w / 2.0, y + h / 2.0))
  return cv.resize(win.astype(np.float32), (PATCH, PATCH), interpolation=cv.INTER_AREA)


def polar_spectrum(patch):
  """Log-polar resampling of the log Fourier magnitude, each radius normalised
  across angles, then unit norm."""
  p = (patch - patch.mean()) * _HANN
  mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(p)))).astype(np.float32)
  c = PATCH / 2.0
  # rows = angle, columns = log radius
  lp = cv.warpPolar(mag, (N_RADIUS, N_ANGLE), (c, c), c,
                    cv.WARP_POLAR_LOG | cv.INTER_LINEAR)
  lp = (lp - lp.mean(axis=0)) / (lp.std(axis=0) + 1e-6)
  n = np.linalg.norm(lp)
  return lp / n if n > 0 else lp


def polar_similarity(a, b):
  """Highest normalised correlation of two polar spectra over circular angle
  shifts and up to MAX_RADIUS_SHIFT log-radius shifts; also the angle shift
  (deg) where it occurs."""
  corr = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
  r = MAX_RADIUS_SHIFT
  corr = np.concatenate([corr[:, :r + 1], corr[:, -r:]], axis=1)
  i, _ = np.unravel_index(np.argmax(corr), corr.shape)
  return float(corr.max()), 360.0 * i / N_ANGLE


def run_fft_layer(gray, clusters, min_similarity=MIN_SIMILARITY, tie_margin=TIE_MARGIN):
  """
  clusters = {
    "h_matrix": matrix,
    "kp_pairs": (inl_src, inl_dst),
    "match_indices": inlier_edge_idx,
    "inlier_count": int(n_inliers),
    "src_region": _region(inl_src),
    "dst_region": _region(inl_dst),
  }

  `gray` is the image the features saw (after preprocessing). Adds
  "fft_similarity" to every cluster. Returns (best, suspicious): the index of
  the first passing cluster in RANSAC order (None when none passes) and the
  indices of every passing cluster within tie_margin of it, best included.
  """
  for c in clusters:
    a = polar_spectrum(side_patch(gray, c["src_region"]))
    b = polar_spectrum(side_patch(gray, c["dst_region"]))
    c["fft_similarity"] = polar_similarity(a, b)[0]

  passing = [i for i, c in enumerate(clusters) if c["fft_similarity"] >= min_similarity]
  if not passing:
    return None, []
  best = passing[0]
  top = clusters[best]["fft_similarity"]
  suspicious = [i for i in passing if abs(top - clusters[i]["fft_similarity"]) <= tie_margin]
  return best, suspicious
