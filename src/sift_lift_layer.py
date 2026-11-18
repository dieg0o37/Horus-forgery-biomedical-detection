import os, sys, csv, numpy as np, shutil, cv2 as cv

# LIFT's dependencies (torch, h5py) are imported in lift.py, only when LIFT is
# used, so SIFT runs still work in an environment without torch.

# util/ sits next to src/, so make the repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.region_from_kp import _region
# neighbours fetched per query before the min_distance filter. Must be large
# enough that >=2 candidates survive that filter; diagnose_sift.py reports a
# 'knn_k_truncation' bucket if it ever is not.
KNN_K = 10

# RANSAC clustering, shared by run_sift_layer and diagnose_sift.py
RANSAC_MAX_CLUSTERS = 100
# 2 px: on the full test split it finds the same 1,860/2,122 images as 5 px, with
# more accurate transforms (1,839 vs 1,822) and 11% fewer false clusters; copies
# shift pixels exactly, so a loose threshold only lets RANSAC bend the model
RANSAC_THRESHOLD = 2.0
# 10: an image is flagged only when cluster #1 has at least 10 inliers. On the
# 2,691 unforged base images it flags 5.9%, and 0.93% on cells, particles, blots
# or background (the rest are labels, rulers, scale bars); it keeps 85.8% of the
# 10,619 forged images (90.7% -> 82.7% on test, against 4 inliers)
RANSAC_MIN_INLIERS = 10

PREPROCESS_MODES = ("none", "stretch", "clahe")
# stretch is the default: on all 10,619 test/val/train images it found 95.3% of
# forgeries vs 93.9% for clahe (88.7% vs none on test), with higher match
# precision and lower runtime. none and clahe stay selectable for comparison.
DEFAULT_PREPROCESS = "stretch"
_CLAHE = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(gray, mode=DEFAULT_PREPROCESS):
  if mode == "none":
    return gray
  if mode == "stretch":
    lo, hi = np.percentile(gray, (0.5, 99.5))
    if hi - lo < 1:
      return gray                     # flat image: nothing to stretch
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)
  if mode == "clahe":
    return _CLAHE.apply(gray)
  raise ValueError(f"unknown preprocess mode {mode!r}, expected one of {PREPROCESS_MODES}")

FEATURE_METHODS = ("sift", "lift")


def make_feature_extractor(method="sift", contrast_thresh=0.01, edge_thresh=30.0,
                           lift_max_keypoints=None, lift_device=None, lift_model_dir=None):
  """A function gray -> (keypoints, descriptors), the contract of
  cv.SIFT.detectAndCompute, for either feature method. Build it once and reuse
  it: LIFT loads its network weights here."""
  if method == "sift":
    sift = cv.SIFT_create(contrastThreshold=contrast_thresh, edgeThreshold=edge_thresh)
    return lambda gray: sift.detectAndCompute(gray, None)
  if method == "lift":
    from lift import LIFT, DEFAULT_MAX_KEYPOINTS
    net = LIFT(model_dir=lift_model_dir, device=lift_device,
               max_keypoints=DEFAULT_MAX_KEYPOINTS if lift_max_keypoints is None else lift_max_keypoints)
    return net.detect_and_describe
  raise ValueError(f"unknown feature method {method!r}, expected one of {FEATURE_METHODS}")


def keep_one_direction(match_idx):
  return [(q, t) for q, t in match_idx if q > t]


def find_good_matches(kp, descriptors, ratio_threshold, min_distance, one_direction=True):
  if (descriptors is None) or len(descriptors) <= 2:
    return []

  bf = cv.BFMatcher()

  raw_matches = bf.knnMatch(descriptors, descriptors, k=min(KNN_K, len(descriptors)))

  pts = np.array([k.pt for k in kp], dtype=np.float64)
  tid = np.array([[m.trainIdx for m in ms] for ms in raw_matches], dtype=np.intp)
  dist = np.array([[m.distance for m in ms] for ms in raw_matches], dtype=np.float64)
  qs = np.arange(len(raw_matches), dtype=np.intp)[:, None]

  pixel_dist = np.linalg.norm(pts[tid] - pts[qs], axis=2) 
  keep = (tid != qs) & (pixel_dist > min_distance)

  rank = np.cumsum(keep, axis=1)                          
  has_two = keep.sum(axis=1) >= 2
  first = np.argmax(keep & (rank == 1), axis=1)
  second = np.argmax(keep & (rank == 2), axis=1)

  rows = np.arange(len(raw_matches), dtype=np.intp)
  d1, d2 = dist[rows, first], dist[rows, second]
  accept = has_two & (d1 < ratio_threshold * d2)

  matches = list(zip(rows[accept].tolist(), tid[rows, first][accept].tolist()))
  return keep_one_direction(matches) if one_direction else matches


def run_RANSAC(good_matches, max_clusters, ransac_threshold, min_inliers):
  n_matches = len(good_matches)

  clusters = []
  sampling_mask = np.ones(n_matches, dtype=bool)

  for _ in range(max_clusters):
    indices = np.where(sampling_mask)[0]

    if len(indices) < min_inliers:
      break

    pts1=np.float64([good_matches[i][0].pt for i in indices])
    pts2=np.float64([good_matches[i][1].pt for i in indices])

    matrix, inlier_mask = cv.estimateAffinePartial2D(
      pts1, pts2, method=cv.RANSAC, ransacReprojThreshold=ransac_threshold
    )

    if matrix is None:
      break

    inlier_mask = inlier_mask.ravel().astype(bool)
    n_inliers = inlier_mask.sum()

    if n_inliers < min_inliers:
      break

    inlier_edge_idx = indices[inlier_mask]
    sampling_mask[inlier_edge_idx] = False

    inl_src = pts1[inlier_mask]
    inl_dst = pts2[inlier_mask]

    clusters.append({
      "h_matrix": matrix,
      "kp_pairs": (inl_src, inl_dst),
      "match_indices": inlier_edge_idx,
      "inlier_count": int(n_inliers),
      "src_region": _region(inl_src), # convex hull = {"hull": ..., "bbox": ...}
      "dst_region": _region(inl_dst),
    })

  return clusters



def run_lift_layer(
      image_folder_path: str,
      output_csv: str,
      image_output_folder: str,
      failed_image_output_folder: str,

      # Hyperparams
      _print: bool,
      ratio_threshold: float,
      min_distance: float,
      preprocess_mode: str = DEFAULT_PREPROCESS,
      use_cuda: bool = None,          # None: CUDA, else Apple GPU, else CPU
      max_keypoints: int = None,      # None: lift.DEFAULT_MAX_KEYPOINTS; 0: no cap
      ) -> None:
    device = None if use_cuda is None else ("cuda" if use_cuda else "cpu")
    extract = make_feature_extractor("lift", lift_max_keypoints=max_keypoints,
                                     lift_device=device)
    _run_layer(extract, image_folder_path, output_csv, image_output_folder,
               failed_image_output_folder, _print, ratio_threshold, min_distance,
               preprocess_mode)


def run_sift_layer(
      image_folder_path: str, 
      output_csv: str, 
      image_output_folder: str,
      failed_image_output_folder: str,
      

      # Hyperparams
      _print: bool,
      ratio_threshold: float, 
      min_distance: float,
      contrast_thresh: float, 
      edge_thresh: float,
      preprocess_mode: str = DEFAULT_PREPROCESS
      ) -> None:
    extract = make_feature_extractor("sift", contrast_thresh, edge_thresh)
    _run_layer(extract, image_folder_path, output_csv, image_output_folder,
               failed_image_output_folder, _print, ratio_threshold, min_distance,
               preprocess_mode)


def _run_layer(extract, image_folder_path, output_csv, image_output_folder,
               failed_image_output_folder, _print, ratio_threshold, min_distance,
               preprocess_mode):
    if _print:
      os.makedirs(image_output_folder, exist_ok=True)
      os.makedirs(failed_image_output_folder, exist_ok=True)

    imgs = os.listdir(image_folder_path)
    valid_endings = ['.png', '.tiff', '.tif', '.jpg', '.jpeg']
    imgs = [i for i in imgs if any(i.endswith(end) for end in valid_endings)]


    with open(output_csv, encoding="utf-8", newline="", mode="w") as f:
      writer = csv.writer(f)

      headers = [
        'image_name',
        'total_keypoints_found',
        'suspicious_kp_pairs_found',
        'suspicious_cluster_pairs_found',
      ]

      writer.writerow(headers)

      for idx, img_name in enumerate(sorted(imgs)):
        img_path = os.path.join(image_folder_path, img_name)

        img = cv.imread(img_path)

        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        kp, descriptors = extract(preprocess(gray_img, preprocess_mode))

        if _print:
            kp_img = cv.drawKeypoints(image=gray_img, keypoints=kp, outImage=None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv.imwrite(os.path.join(image_output_folder, f'{idx}.jpg'), kp_img)

        n_kp = len(kp)

        match_idx = find_good_matches(kp, descriptors, ratio_threshold, min_distance)
        good_matches = [(kp[qi], kp[ti]) for qi, ti in match_idx]
        sus_match_count = len(good_matches)

        clusters = run_RANSAC(good_matches=good_matches, max_clusters=RANSAC_MAX_CLUSTERS,
                              ransac_threshold=RANSAC_THRESHOLD, min_inliers=RANSAC_MIN_INLIERS)
        n_clusters = len(clusters)

        if n_clusters==0 and _print:
            kp_img = cv.drawKeypoints(image=gray_img, keypoints=kp, outImage=None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv.imwrite(os.path.join(failed_image_output_folder, f'{idx}.jpg'), kp_img)


        writer.writerow([img_name, n_kp, sus_match_count, n_clusters])
           
      


if __name__ == "__main__":
  output_csv="../logs/sift_logs.csv"
  image_folder_path="../data/real_samples/images/"
  image_output_folder="../results/sift_layer/"
  failed_image_output_folder="../results/sift_layer/no_clusters/"
  import sys
  input=sys.argv[1::]

  try:
    if input[0]=="True":
       _print = True
    elif input[0]=="False":
       _print=False
    else:
      print("input format: python sift_lift_layer.py <_print> <ratio_threshold> <min_distance> <contrast_threshold> <edge_threshold> [none|stretch|clahe] [sift|lift] (defaults stretch, sift; contrast/edge only apply to sift)")
      exit(1)
    ratio_threshold = float(input[1])
    min_dist = float(input[2])
    contrast_threshold = float(input[3])
    edge_threshold = float(input[4])
    preprocess_mode = input[5] if len(input) > 5 else DEFAULT_PREPROCESS
    features = input[6] if len(input) > 6 else "sift"
    if preprocess_mode not in PREPROCESS_MODES or features not in FEATURE_METHODS:
      print("input format: python sift_lift_layer.py <_print> <ratio_threshold> <min_distance> <contrast_threshold> <edge_threshold> [none|stretch|clahe] [sift|lift] (defaults stretch, sift; contrast/edge only apply to sift)")
      exit(1)
    if features == "lift":
      run_lift_layer(image_folder_path=image_folder_path, output_csv=output_csv,
                     image_output_folder=image_output_folder,
                     failed_image_output_folder=failed_image_output_folder,
                     _print=_print, ratio_threshold=ratio_threshold,
                     min_distance=min_dist, preprocess_mode=preprocess_mode)
      exit(0)
    run_sift_layer(
       image_folder_path=image_folder_path,
       output_csv=output_csv,
       image_output_folder=image_output_folder,
       failed_image_output_folder=failed_image_output_folder,

       _print=_print,
       ratio_threshold=ratio_threshold,
       min_distance=min_dist,
       contrast_thresh=contrast_threshold,
       edge_thresh=edge_threshold,
       preprocess_mode=preprocess_mode
    )
    # clusters = run_RANSAC(good_matches=good_matches, max_clusters=100, ransac_threshold=5.0, min_inliers=4)
  except TypeError:
     print("input format: python sift_lift_layer.py <_print> <ratio_threshold> <min_distance> <contrast_threshold> <edge_threshold> [none|stretch|clahe] [sift|lift] (defaults stretch, sift; contrast/edge only apply to sift)")
     exit(1)
      








