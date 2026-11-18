"""LIFT keypoints and descriptors (Yi, Trulls, Lepetit, Fua, ECCV 2016) in PyTorch.

A port of the test-time path of the authors' release, cvlab-epfl/LIFT
(compute_detector.py, compute_orientation.py, compute_descriptor.py), running
their pretrained weights (models/picc-best and models/base). The original runs
on Theano/Lasagne, which no longer installs on current Python. The software is
patented and released for academic use only.

  detector     one 25x25 conv layer (16 filters) + GHH max/sum pooling, applied
               to the image resized for 21 scales; 2-D and scale-space
               non-maximum suppression, an edge test and sub-pixel refinement
  orientation  3 conv + 2 dense layers on a 64x64 patch of radius 6 x scale,
               output angle = atan2(o0, o1)
  descriptor   3 conv / tanh / L2-pooling / subtractive-normalisation blocks on
               the same patch rotated by that angle, 128 floats, compared with L2

Only resampling differs from the original: orientation and descriptor patches
are sampled from the image in one bilinear step, where the original first warps
a 128x128 uint8 patch with cv2.warpAffine and then resamples its centre with a
spatial transformer. The geometry (centre, radius, rotation) is identical.
"""
import os, urllib.request

import numpy as np, cv2 as cv
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.environ.get(
  "LIFT_MODEL_DIR", os.path.join(REPO_ROOT, "Lift_model", "LIFT", "models"))
WEIGHT_FILES = ("picc-best/model.h5", "picc-best/mean_std.h5",
                "base/new-CNN3-picc-iter-56k.h5")
WEIGHT_URL = "https://github.com/cvlab-epfl/LIFT/raw/master/models/"

# models/configs/picc-finetune-nopair.config and the compute_* scripts
FILTER_SIZE = 25                       # detector filter, model.nFilterSize
PATCH_SIZE_KP = 48                     # model.nPatchSizeKp
DESC_INPUT = 64                        # model.nDescInputSize
RATIO_SCALE = (PATCH_SIZE_KP / 2.0) / 2.0      # 12: big-patch radius = 12 x scale
SCL_INTV, NMS_INTV, EDGE_TH = 4, 2, 10.0       # validation defaults in compute_detector
MIN_SCALE_LOG2, MAX_SCALE_LOG2 = 1, 4
# NMS window: round(0.5 * (48 - 1) * 64 / 128) = 12, times 0.25 -> 3, a 7x7 window
NEARBY = max(int(np.round(int(np.round(0.5 * (PATCH_SIZE_KP - 1.0) * DESC_INPUT / 128.0))
                          * 0.25)), 1)
# the original keeps the 1000 best; copy-move needs many more (SIFT gives
# thousands on the dense datasets), so the default cap is higher
DEFAULT_MAX_KEYPOINTS = 5000
GHH_SUM, GHH_MAX = 4, 4


def ensure_weights(model_dir=DEFAULT_MODEL_DIR):
  """Download the three weight files from the authors' repository if missing."""
  for rel in WEIGHT_FILES:
    path = os.path.join(model_dir, rel)
    if os.path.isfile(path):
      continue
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading LIFT weights {rel} -> {path}", flush=True)
    tmp = path + ".part"
    urllib.request.urlretrieve(WEIGHT_URL + rel, tmp)
    os.replace(tmp, path)
  return model_dir


def pick_device(device=None):
  if device not in (None, "", "auto"):
    return torch.device(device)
  if torch.cuda.is_available():
    return torch.device("cuda")
  if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def _ghh(x, num_out):
  """GHH feature pooling (ghh_pool.make_ghh_pool_conv2d, hard max): channels are
  grouped as (out, sum, max); max over the last, then sum with signs +,-,+,-."""
  shape = x.shape
  x = x.reshape(shape[0], num_out, GHH_SUM, GHH_MAX, *shape[2:]).amax(dim=3)
  delta = torch.tensor([1.0, -1.0, 1.0, -1.0], device=x.device, dtype=x.dtype)
  return (x * delta.view(1, 1, GHH_SUM, *([1] * (x.dim() - 3)))).sum(dim=2)


class LIFT:
  def __init__(self, model_dir=None, device=None, max_keypoints=DEFAULT_MAX_KEYPOINTS,
               batch_size=512):
    import h5py
    model_dir = ensure_weights(model_dir or DEFAULT_MODEL_DIR)
    self.device = pick_device(device)
    self.max_keypoints = max_keypoints
    self.batch_size = batch_size
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(self.device)
    with h5py.File(os.path.join(model_dir, "picc-best", "model.h5"), "r") as m:
      g = lambda name, p: m[name][f"{name}.{p}"][()]
      self.kp_w, self.kp_b = t(g("kp-c0", "W")[..., 0]), t(g("kp-c0", "b"))
      self.ori = [(t(g(n, "W")), t(g(n, "b"))) for n in ("ori-c0", "ori-c1", "ori-c2", "ori-f3", "ori-f4")]
      self.desc = [(t(g(n, "W")), t(g(n, "b"))) for n in ("desc-1-conv", "desc-5-conv", "desc-9-conv")]
    with h5py.File(os.path.join(model_dir, "picc-best", "mean_std.h5"), "r") as m:
      self.mean, self.std = float(m["mean_x"][()]), float(m["std_x"][()])
    with h5py.File(os.path.join(model_dir, "base", "new-CNN3-picc-iter-56k.h5"), "r") as m:
      assert list(m["layers"][()]) == [1, 5, 4, 7, 1, 5, 4, 7, 1, 5, 4], "unexpected descriptor layout"
      self.desc_mean, self.desc_std = float(m["patch-mean"][0]), float(m["patch-std"][0])
      self.desc_pool = [int(m[f"l-{i}-kw"][0]) for i in (3, 7, 11)]
      self.desc_norm = [t(m[f"l-{i}-filter"][()]) for i in (4, 8)]
    u = torch.linspace(-1.0, 1.0, DESC_INPUT, device=self.device)
    self._gy, self._gx = torch.meshgrid(u, u, indexing="ij")    # spatial-transformer grid

  # ------------------------------------------------------------------ detector

  @torch.no_grad()
  def _score_map(self, img):
    """Detector response of one resized image, padded back to its size with -inf
    (filter_tools.apply_learned_filter_2_image_no_theano + the -inf padding)."""
    h, w = img.shape
    half = FILTER_SIZE // 2
    if h <= 2 * half or w <= 2 * half:
      return np.full((h, w), -np.inf, np.float32)
    x = torch.from_numpy(((img - self.mean) / self.std).astype(np.float32)).to(self.device)
    out = torch.empty((h - 2 * half, w - 2 * half), device=self.device)
    rows = max(16, int(2e6 // max(w, 1)))         # output rows per strip: bounds the 16-channel buffer
    for r0 in range(0, out.shape[0], rows):
      r1 = min(r0 + rows, out.shape[0])
      strip = x[r0:r1 + 2 * half][None, None]
      resp = F.conv2d(strip, self.kp_w, self.kp_b)          # (1, 16, rows, w - 24)
      out[r0:r1] = _ghh(resp, 1)[0, 0]
    return np.pad(out.cpu().numpy(), half, mode="constant", constant_values=-np.inf)

  @torch.no_grad()
  def _nms2d(self, score):
    """kp_tools.nonMaxSuppression, 2-D part: strict maximum of its 7x7 window,
    with the whole window inside the valid (finite) score map."""
    k = 2 * NEARBY + 1
    s = torch.from_numpy(score).to(self.device)
    pad = F.pad(s[None, None], (NEARBY,) * 4, value=float("-inf"))
    mx = F.max_pool2d(pad, k, stride=1)[0, 0]
    mn = -F.max_pool2d(-pad, k, stride=1)[0, 0]
    # max > second largest  <=>  the max occurs once in the window
    count = torch.zeros_like(s)
    H, W = s.shape
    for dy in range(k):
      for dx in range(k):
        count += (pad[0, 0, dy:dy + H, dx:dx + W] == mx)
    keep = (s == mx) & (count == 1) & torch.isfinite(mn)
    return keep.cpu().numpy(), mx.cpu().numpy()

  def _subpixel(self, scores, nms, resizes, scales):
    """kp_tools.get_subpixel_XYZS: 3-D quadratic fit in (x, y, scale), edge test,
    offsets under 1.5 in every axis. Returns rows of (x, y, scale, score)."""
    log_step = (np.log2(scales).max() - np.log2(scales).min()) / (len(scales) - 1.0)
    out = []
    for i, mask in enumerate(nms):
      py, px = np.nonzero(mask)
      if len(py) == 0:
        continue
      ratio = {d: (np.asarray(scores[i + d].shape, float) - 1.0) / (np.asarray(scores[i].shape, float) - 1.0)
               for d in (-1, 1)}
      ratio[0] = np.ones(2)

      def at(dx, dy, ds, py=py, px=px):
        dx, dy, ds = (np.broadcast_to(np.asarray(v, float), py.shape) for v in (dx, dy, ds))
        ds = np.round(ds).astype(int)
        res = np.empty(py.shape)
        for d in (-1, 0, 1):
          m = ds == d
          if not m.any():
            continue
          r, sc = ratio[d], scores[i + d]
          yy = np.clip(np.round((py[m] + dy[m]) * r[0]).astype(int), 0, sc.shape[0] - 1)
          xx = np.clip(np.round((px[m] + dx[m]) * r[1]).astype(int), 0, sc.shape[1] - 1)
          res[m] = sc[yy, xx]
        return res

      with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        c = at(0, 0, 0)
        Dx = 0.5 * (at(1, 0, 0) - at(-1, 0, 0))
        Dy = 0.5 * (at(0, 1, 0) - at(0, -1, 0))
        Ds = 0.5 * (at(0, 0, 1) - at(0, 0, -1))
        Dxx = at(1, 0, 0) + at(-1, 0, 0) - 2.0 * c
        Dyy = at(0, 1, 0) + at(0, -1, 0) - 2.0 * c
        Dss = at(0, 0, 1) + at(0, 0, -1) - 2.0 * c
        Dxy = 0.25 * (at(1, 1, 0) + at(-1, -1, 0) - at(-1, 1, 0) - at(1, -1, 0))
        Dxs = 0.25 * (at(1, 0, 1) + at(-1, 0, -1) - at(-1, 0, 1) - at(1, 0, -1))
        Dys = 0.25 * (at(0, 1, 1) + at(0, -1, -1) - at(0, -1, 1) - at(0, 1, -1))
        D = np.stack([Dx, Dy, Ds, Dxx, Dyy, Dss, Dxy, Dxs, Dys])
        good = np.isfinite(D).all(axis=0)
        edge = (Dxx + Dyy) ** 2 / (Dxx * Dyy - Dxy ** 2)
        good &= (edge >= 0) & (edge < (EDGE_TH + 1.0) ** 2 / EDGE_TH)
      if not good.any():
        continue
      py, px, D = py[good], px[good], D[:, good]
      Dx, Dy, Ds, Dxx, Dyy, Dss, Dxy, Dxs, Dys = D
      A = np.stack([np.stack([Dxx, Dxy, Dxs], -1), np.stack([Dxy, Dyy, Dys], -1),
                    np.stack([Dxs, Dys, Dss], -1)], 1)
      b = -np.stack([Dx, Dy, Ds], -1)
      sol = np.full_like(b, np.nan)
      ok = np.abs(np.linalg.det(A)) > 1e-12      # lu_solve on a singular A gives inf/nan
      if ok.any():
        sol[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
      keep = (np.abs(sol) < 1.5).all(axis=1)
      if not keep.any():
        continue
      sol, py, px = sol[keep], py[keep], px[keep]
      score = at(sol[:, 0], sol[:, 1], sol[:, 2], py=py, px=px)
      out.append(np.stack([(px + sol[:, 0]) / resizes[i], (py + sol[:, 1]) / resizes[i],
                           scales[i] * 2.0 ** (sol[:, 2] * log_step), score], 1))
    return np.concatenate(out) if out else np.zeros((0, 4))

  def detect(self, gray):
    """Keypoints of a grey image as rows of (x, y, scale, score), best first."""
    gray = np.asarray(gray, np.float32)
    h, w = gray.shape
    lo = MIN_SCALE_LOG2 - (1 if min(h, w) <= 1600 else 0)      # double scale for small images
    scales = 2.0 ** np.linspace(lo, MAX_SCALE_LOG2, (MAX_SCALE_LOG2 - lo) * (SCL_INTV + 1) + 1)
    resizes = ((PATCH_SIZE_KP - 1) / 2.0) / (RATIO_SCALE * scales)
    valid = resizes * min(h, w) > FILTER_SIZE + 1
    if not valid.all():
      first = int(np.nonzero(~valid)[0][0])
      scales, resizes = scales[:first], resizes[:first]
    if len(scales) < 2 * NMS_INTV + 1:
      return np.zeros((0, 4))
    scores = [self._score_map(cv.resize(gray, (int(np.round(w * r)), int(np.round(h * r)))))
              for r in resizes]
    nms2d = [self._nms2d(s) for s in scores]
    masks = []
    for i, (keep, mx) in enumerate(nms2d):
      mask = np.zeros_like(keep)
      if NMS_INTV <= i < len(scores) - NMS_INTV:       # needs both neighbouring scales
        cy, cx = np.nonzero(keep)
        for d in (-2, -1, 1, 2):
          other = nms2d[i + d][1]
          r = (np.asarray(other.shape, float) - 1.0) / (np.asarray(mx.shape, float) - 1.0)
          oy, ox = np.round(cy * r[0]).astype(int), np.round(cx * r[1]).astype(int)
          good = (mx[cy, cx] > other[oy, ox]) & np.isfinite(other[oy, ox])
          cy, cx = cy[good], cx[good]
        mask[cy, cx] = True
      masks.append(mask)
    xyzs = self._subpixel(scores, masks, resizes, scales)
    xyzs = xyzs[np.argsort(xyzs[:, 3])[::-1]]
    return xyzs[:self.max_keypoints] if self.max_keypoints else xyzs

  # ------------------------------------------------------- orientation and descriptor

  def _patches(self, img, xy, scale, angle=None):
    """64x64 patches of radius 6 x scale (the centre half of the original's
    128x128, radius-12 x scale crop), rotated by `angle` (radians)."""
    n = len(xy)
    r = (0.5 * RATIO_SCALE * scale).view(n, 1, 1)                 # 6 x scale
    gx, gy = self._gx[None], self._gy[None]
    if angle is not None:
      c, s = torch.cos(angle).view(n, 1, 1), torch.sin(angle).view(n, 1, 1)
      gx, gy = c * gx - s * gy, s * gx + c * gy
    X = xy[:, 0].view(n, 1, 1) + r * gx
    Y = xy[:, 1].view(n, 1, 1) + r * gy
    H, W = img.shape[-2:]
    grid = torch.stack([2.0 * X / (W - 1) - 1.0, 2.0 * Y / (H - 1) - 1.0], dim=-1)
    return F.grid_sample(img.expand(n, 1, H, W), grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)

  def _orientation(self, patch):
    x = (patch - self.mean) / self.std
    (w0, b0), (w1, b1), (w2, b2), (w3, b3), (w4, b4) = self.ori
    x = F.max_pool2d(F.relu(F.conv2d(x, w0, b0)), 2)
    x = F.max_pool2d(F.relu(F.conv2d(x, w1, b1)), 2)
    x = F.relu(F.conv2d(x, w2, b2))           # ori-f3 reads ori-c2a, before pooling
    x = _ghh(x.flatten(1) @ w3 + b3, 100)
    x = _ghh(x @ w4 + b4, 2)
    return torch.atan2(x[:, 0], x[:, 1])

  def _subtractive_norm(self, x, kernel):
    """SubtractiveNormalization2DLayer: subtract a local mean taken over all
    channels with the normalised kernel, corrected at the borders."""
    k = kernel / kernel.sum()
    c, pad = x.shape[1], k.shape[-1] // 2
    mean = F.conv2d(x, (k / c).expand(1, c, *k.shape), padding=pad)
    coef = F.conv2d(torch.ones(1, 1, *x.shape[-2:], device=x.device), k[None, None], padding=pad)
    return x - mean / coef

  def _descriptor(self, patch):
    x = (patch - self.desc_mean) / self.desc_std
    for j, (w, b) in enumerate(self.desc):
      x = torch.tanh(F.conv2d(x, w, b))
      x = F.lp_pool2d(x, 2, self.desc_pool[j])
      if j < 2:
        x = self._subtractive_norm(x, self.desc_norm[j])
    return x.flatten(1)

  @torch.no_grad()
  def describe(self, gray, xyzs):
    """Orientation and 128-d descriptor for each keypoint. Like the original
    (dataset_tools.helper.load_patches), keypoints whose big crop leaves the
    image are dropped; returns (kept rows, angles in radians, descriptors)."""
    gray = np.asarray(gray, np.float32)
    H, W = gray.shape
    hw = xyzs[:, 2] * RATIO_SCALE * np.sqrt(2.0)
    xs, xe = np.round(xyzs[:, 0] - hw), np.round(xyzs[:, 0] + hw)
    ys = np.round(xyzs[:, 1] - hw)
    # the original tests ys twice and never ye (helper.py); kept as released
    keep = (xs >= 1) & (xe < W - 1) & (ys >= 1) & (ys < H - 1)
    xyzs = xyzs[keep]
    if len(xyzs) == 0:
      return xyzs, np.zeros(0), np.zeros((0, 128), np.float32)
    img = torch.from_numpy(gray).to(self.device)[None, None]
    angles, descs = [], []
    for s in range(0, len(xyzs), self.batch_size):
      chunk = torch.from_numpy(xyzs[s:s + self.batch_size].astype(np.float32)).to(self.device)
      xy, scale = chunk[:, :2], chunk[:, 2]
      ang = self._orientation(self._patches(img, xy, scale))
      descs.append(self._descriptor(self._patches(img, xy, scale, ang)).cpu().numpy())
      angles.append(ang.cpu().numpy())
    return xyzs, np.concatenate(angles), np.concatenate(descs).astype(np.float32)

  def detect_and_describe(self, gray):
    """Same contract as cv.SIFT.detectAndCompute: (list of cv.KeyPoint, N x 128
    float32 descriptors or None). KeyPoint.size is the descriptor support
    diameter (12 x scale), angle in degrees, response the detector score."""
    xyzs = self.detect(gray)
    xyzs, angles, desc = self.describe(gray, xyzs)
    # order by x (then y), as OpenCV returns SIFT keypoints: the pipeline's
    # keep_one_direction (q > t) relies on that for a consistent direction
    order = np.lexsort((xyzs[:, 1], xyzs[:, 0]))
    xyzs, angles, desc = xyzs[order], angles[order], desc[order]
    kps = [cv.KeyPoint(float(x), float(y), float(2 * 0.5 * RATIO_SCALE * z),
                       float(np.degrees(a) % 360.0), float(sc))
           for (x, y, z, sc), a in zip(xyzs, angles)]
    return kps, (desc if len(kps) else None)
