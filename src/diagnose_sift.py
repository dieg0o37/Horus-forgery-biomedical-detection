"""Stage-by-stage diagnosis of why the SIFT layer misses a known copy-move forgery.

The SIFT layer can only find a forgery if four things happen in order:

  1. DETECTION      the same physical point is detected in BOTH copies, so a
                    correct pair of keypoints exists at all.
  2. DESCRIPTION    for such a pair, the true partner is the nearest neighbour in
                    descriptor space (the pipeline only ever inspects m1 = NN1).
  3. RATIO TEST     NN1 is enough better than NN2 to pass m1 < ratio * m2.
  4. MIN_DISTANCE   the surviving pair is farther apart than min_distance px.

Usage:
  conda activate torch
  python src/diagnose_sift.py
  python src/diagnose_sift.py --ratio-threshold 0.9 --contrast-thresh 0.005
  python src/diagnose_sift.py --split test --preprocess clahe
"""

import argparse, collections, csv, faulthandler, hashlib, os, random, shutil, signal, socket, sys, time
import numpy as np, cv2 as cv

from sift_layer import (DEFAULT_PREPROCESS, KNN_K, PREPROCESS_MODES, RANSAC_MAX_CLUSTERS,
                        RANSAC_MIN_INLIERS, RANSAC_THRESHOLD, find_good_matches,
                        keep_one_direction, preprocess, run_RANSAC)

RATIO_SWEEP = (0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
QUERY_CHUNK = 256   # query rows of the distance matrix held in memory at once
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ path config

def load_paths(paths_file):
  """Parse config/paths.txt: KEY=VALUE per line, '#' comments, '~' expanded.

  Relative values are resolved against the repo root, not the current working
  directory, so the script behaves the same whether slurm launches it from
  ~/Src/Pesquisa-Lift-Test or from src/.
  """
  if not os.path.isfile(paths_file):
    raise SystemExit(
      f"missing {paths_file}\n"
      f"copy config/paths.example.txt to config/paths.txt and fill in the "
      f"dataset locations, or pass --image-folder/--mask-folder explicitly.")

  cfg = {}
  for lineno, raw in enumerate(open(paths_file, encoding="utf-8"), 1):
    line = raw.split("#", 1)[0].strip()
    if not line:
      continue
    if "=" not in line:
      raise SystemExit(f"{paths_file}:{lineno}: expected KEY=VALUE, got {raw.strip()!r}")
    key, value = line.split("=", 1)
    value = os.path.expanduser(value.strip())
    if value and not os.path.isabs(value):
      value = os.path.normpath(os.path.join(REPO_ROOT, value))
    cfg[key.strip()] = value
  return cfg


def resolve_split(split, paths_file):
  """Look up the image/mask folders for a dataset split in config/paths.txt."""
  cfg = load_paths(paths_file)
  img_key, mask_key = f"CMFD_{split.upper()}_IMAGES", f"CMFD_{split.upper()}_MASKS"
  missing = [k for k in (img_key, mask_key) if not cfg.get(k)]
  if missing:
    raise SystemExit(f"{paths_file} has no value for: {', '.join(missing)}\n"
                     f"keys found: {', '.join(sorted(cfg)) or '(none)'}")
  for key in (img_key, mask_key):
    if not os.path.isdir(cfg[key]):
      raise SystemExit(f"{key} points at {cfg[key]!r}, which is not a directory")
  return cfg[img_key], cfg[mask_key]


# ---------------------------------------------------------------- ground truth

def split_mask(mask):
  """RED channel -> pasted region, GREEN channel -> source region."""
  b, g, r = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]
  dst = (r > 127) & (g < 128) & (b < 128)   # red   = forged / pasted
  src = (g > 127) & (r < 128) & (b < 128)   # green = original source
  return src, dst


def _centroid(m):
  ys, xs = np.nonzero(m)
  return np.array([xs.mean(), ys.mean()])


def _orientation(m):
  """Principal-axis angle from second-order central moments."""
  ys, xs = np.nonzero(m)
  x, y = xs - xs.mean(), ys - ys.mean()
  mu20, mu02, mu11 = (x * x).mean(), (y * y).mean(), (x * y).mean()
  return 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)


def _warp(mask, matrix, shape):
  return cv.warpAffine(mask.astype(np.uint8), matrix, (shape[1], shape[0]),
                       flags=cv.INTER_NEAREST) > 0


def _shift_by_overlap(src, dst):
  """Integer shift t that maximises |(src + t) & dst|, the number of pasted-region
  pixels explained by the shifted source, by FFT cross-correlation of the two
  masks' bounding-box crops (zero-padded, so there is no wrap-around).

  Unlike the centroid difference this stays exact when the two regions overlap
  and the mask paints the overlap in only one colour: the hidden part shifts that
  region's centroid, but at the true shift every visible pixel of the smaller
  mask is still covered, which is the largest overlap any shift can reach.
  """
  ys, xs = np.nonzero(src)
  yd, xd = np.nonzero(dst)
  a = src[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
  b = dst[yd.min():yd.max() + 1, xd.min():xd.max() + 1].astype(np.float32)
  h, w = a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1
  corr = np.fft.irfft2(np.fft.rfft2(b, (h, w)) * np.conj(np.fft.rfft2(a, (h, w))), (h, w))
  ky, kx = np.unravel_index(np.argmax(np.rint(corr)), corr.shape)
  ky = ky if ky < b.shape[0] else ky - h        # indices past b's size are negative lags
  kx = kx if kx < b.shape[1] else kx - w
  return xd.min() - xs.min() + kx, yd.min() - ys.min() + ky


def _fit(a, b, matrix):
  """How well `matrix` maps mask a onto mask b, allowing for the overlap.

  recall    = share of b covered by the warped a
  precision = share of the warped a that lands on b, or on a itself: where the
              copy overlaps its source, the mask shows only one of the two
              colours, so the warped a legitimately lands on a there.
  """
  wa = _warp(a, matrix, b.shape)
  n_wa, n_b = np.count_nonzero(wa), np.count_nonzero(b)
  if not n_wa or not n_b:
    return 0.0
  recall = np.count_nonzero(wa & b) / n_b
  precision = np.count_nonzero(wa & (a | b)) / n_wa
  return float(min(recall, precision))


def _gt_score(src, dst, matrix):
  # whichever mask lost the overlap, one of the two directions sees the full region
  return max(_fit(src, dst, matrix), _fit(dst, src, cv.invertAffineTransform(matrix)))


def estimate_gt_transform(src, dst):
  """
  Tries pure translation first and only falls back to a moment-based similarity search if
  translation fails to explain the masks. Returns (matrix, method, score), where score is
  the overlap-aware fit of _gt_score (1.0 = the transform explains both masks exactly).

  The translation is the shift with the largest mask overlap (_shift_by_overlap), not the
  centroid difference: when the source and the copy overlap, the mask paints the shared
  pixels in one colour only, which moves one centroid by several pixels. That put the
  ground truth 3-6px off on ~12% of the images first reported as failures, so correct
  matches fell outside --tol-px and were counted as wrong.
  """
  cs, cd = _centroid(src), _centroid(dst)

  best = (None, "translation", -1.0)
  shifts = (_shift_by_overlap(src, dst), (cd[0] - cs[0], cd[1] - cs[1]))
  for tx, ty in shifts:
    trans = np.float32([[1, 0, tx], [0, 1, ty]])
    score = _gt_score(src, dst, trans)
    if score > best[2]:
      best = (trans, "translation", score)

  if best[2] < 0.9:
    scale = np.sqrt(np.count_nonzero(dst) / max(np.count_nonzero(src), 1))
    dtheta = _orientation(dst) - _orientation(src)
    # the principal axis is only defined up to 180 deg, and moment estimates are
    # coarse, so search both branches plus a small offset grid
    for branch in (0.0, np.pi):
      for extra in np.deg2rad(np.arange(-8.0, 8.1, 2.0)):
        th = dtheta + branch + extra
        rot = scale * np.array([[np.cos(th), -np.sin(th)],
                                [np.sin(th), np.cos(th)]])
        off = cd - rot @ cs
        cand = np.float32(np.hstack([rot, off.reshape(2, 1)]))
        score = _gt_score(src, dst, cand)
        if score > best[2]:
          best = (cand, "similarity", score)

  return best


# ------------------------------------------------------------------ diagnostic

def _l2_rows(des, rows):
  des = np.ascontiguousarray(des, dtype=np.float32)
  sq = (des * des).sum(axis=1)
  d2 = sq[rows][:, None] + sq[None, :] - 2.0 * (des[rows] @ des.T)
  np.maximum(d2, 0.0, out=d2)
  return np.sqrt(d2, out=d2)


def _two_smallest(d):
  """Indices of the two smallest entries of `d`, lowest index first on ties.

  np.argmin returns the first occurrence, so this matches
  np.argsort(kind="stable")[:2] exactly, but costs O(n) instead of O(n log n).
  At 23k keypoints the two full sorts this replaces were ~3ms per query, which
  dominated the whole diagnostic.
  """
  i1 = int(np.argmin(d))
  keep = d[i1]
  d[i1] = np.inf
  i2 = int(np.argmin(d)) if len(d) > 1 else None
  d[i1] = keep
  if i2 is not None and not np.isfinite(d[i2]):
    i2 = None
  return i1, i2


CLUSTER_DEFAULTS = {
  "n_matches_kept": 0, "n_matches_kept_correct": 0,
  "n_clusters": 0, "n_clusters_correct": 0, "n_clusters_false": 0,
  "cluster_found": 0, "cluster_mirrored": 0,
  # rows that return before clustering (no keypoints) have no correct match
  "cluster_outcome": "no_correct_match",
  "first_correct_rank": "", "best_cluster_inliers": "",
  "best_cluster_coverage": "", "best_cluster_purity": "",
  "best_cluster_transform_err": "",
}


def score_clusters(clusters, matrix, inv, src_mask, dst_mask, tol_px, row):
  """Label every RANSAC cluster correct or false against the ground truth.

  A cluster is correct when most of its inlier pairs are true correspondences,
  in either direction. `first_correct_rank` is its position in RANSAC order,
  i.e. how many candidates a verifier checking them in order must look at.
  For the largest correct cluster, coverage = share of the true region its
  keypoint hulls cover, purity = share of the hulls inside the true region;
  those hulls are what a region-level verifier (the FFT layer) will receive.
  transform_err = 95th-percentile px distance, over the whole true region,
  between where the cluster's matrix and the true transform send each pixel.
  Measured on the region, not the cluster's own points: RANSAC can bend a
  scale/rotation model around a few false matches so that it still fits its
  inliers yet misplaces the rest of the patch, and a verifier warps the patch.
  """
  row["n_clusters"] = len(clusters)
  best, directions = None, set()
  for rank, c in enumerate(clusters, 1):
    a, b = c["kp_pairs"]
    ah = np.hstack([a, np.ones((len(a), 1), np.float32)])
    fwd = np.mean(np.linalg.norm(ah @ matrix.T - b, axis=1) <= tol_px)
    rev = np.mean(np.linalg.norm(ah @ inv.T - b, axis=1) <= tol_px)
    if max(fwd, rev) < 0.5:
      row["n_clusters_false"] += 1
      continue
    row["n_clusters_correct"] += 1
    directions.add(fwd >= rev)
    if row["first_correct_rank"] == "":
      row["first_correct_rank"] = rank
    if best is None or c["inlier_count"] > best[0]["inlier_count"]:
      best = (c, fwd >= rev)
  # the same forgery fitted in both directions = the duplicate keep_one_direction
  # is meant to prevent (it can still happen for near-vertical copies)
  row["cluster_mirrored"] = int(len(directions) == 2)
  if best is None:
    return
  c, forward = best
  gt_a, gt_b = (src_mask, dst_mask) if forward else (dst_mask, src_mask)
  truth_t = matrix if forward else inv
  ys, xs = np.nonzero(gt_a)
  step = max(1, len(xs) // 4000)                 # subsample big regions
  ph = np.column_stack([xs[::step], ys[::step], np.ones(len(xs[::step]))])
  row["best_cluster_transform_err"] = round(float(np.percentile(
    np.linalg.norm(ph @ c["h_matrix"].T - ph @ truth_t.T, axis=1), 95)), 3)
  cov, pur = [], []
  for region, truth in ((c["src_region"], gt_a), (c["dst_region"], gt_b)):
    hull = np.zeros(truth.shape, np.uint8)
    cv.fillConvexPoly(hull, region["hull"], 1)
    hull = hull.astype(bool)
    inter = np.count_nonzero(hull & truth)
    cov.append(inter / max(np.count_nonzero(truth), 1))
    pur.append(inter / max(np.count_nonzero(hull), 1))
  row.update(cluster_found=1, best_cluster_inliers=c["inlier_count"],
             best_cluster_coverage=round(float(np.mean(cov)), 4),
             best_cluster_purity=round(float(np.mean(pur)), 4))


def diagnose_image(gray, src_mask, dst_mask, gt, args):
  matrix, method, gt_iou = gt
  inv = cv.invertAffineTransform(matrix)

  t0 = time.perf_counter()
  sift = cv.SIFT_create(contrastThreshold=args.contrast_thresh,
                        edgeThreshold=args.edge_thresh)
  kp, des = sift.detectAndCompute(gray, None)
  sec_sift = time.perf_counter() - t0

  row = {
    "gt_method": method,
    "gt_mask_iou": round(gt_iou, 4),   # overlap-aware fit (_gt_score); name kept for old CSVs
    "gt_dx": round(float(matrix[0, 2]), 2),
    "gt_dy": round(float(matrix[1, 2]), 2),
    "src_area_px": int(np.count_nonzero(src_mask)),
    "dst_area_px": int(np.count_nonzero(dst_mask)),
    "n_kp_total": len(kp),
    "sec_sift": round(sec_sift, 4), "sec_match": 0.0, "sec_stats": 0.0,
    "sec_ransac": 0.0,
    **CLUSTER_DEFAULTS,
  }
  counters = ["n_kp_src", "n_kp_dst", "n_queries", "n_queries_total",
              "n_detected", "n_detected_far",
              "n_rank1_far", "n_rank1_all", "n_rank_all_le_knnk",
              "n_pass_ratio_first", "n_pass_dist_first",
              "n_blocked_by_duplicate_partner", "n_blocked_by_duplicate_denominator",
              "n_good_matches", "n_good_matches_correct", "n_correct_pairs_distinct",
              "n_sim_mismatch"]
  row.update({c: 0 for c in counters})
  row.update({f"n_pass_far_{t:g}": 0 for t in RATIO_SWEEP})
  row.update({"repeatability": "", "median_rank_all": "", "median_rank_far": "",
              "median_ratio_global": "", "median_ratio_far": "",
              "median_ratio_vs_nonpartner": "", "lost_at": ""})

  if len(kp) == 0:
    row["lost_at"] = "no_keypoints"
    return row

  pts = np.array([k.pt for k in kp], dtype=np.float32)
  sizes = np.array([k.size for k in kp], dtype=np.float32)
  h, w = gray.shape[:2]
  ix = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
  iy = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
  in_src, in_dst = src_mask[iy, ix], dst_mask[iy, ix]
  row["n_kp_src"], row["n_kp_dst"] = int(in_src.sum()), int(in_dst.sum())

  # --- the pipeline's own output, as the end-to-end truth for this image ------
  t0 = time.perf_counter()
  # both directions: the per-query check below follows every query. The
  # pipeline then keeps one direction per pair (keep_one_direction) for RANSAC.
  match_idx = find_good_matches(kp, des, args.ratio_threshold, args.min_distance,
                                one_direction=False)
  row["sec_match"] = round(time.perf_counter() - t0, 4)
  t_stats = time.perf_counter()
  row["n_good_matches"] = len(match_idx)
  correct_by_query, correct_pairs = set(), set()
  for qi, ti in match_idx:
    if in_src[qi]:
      expected = matrix @ np.array([pts[qi, 0], pts[qi, 1], 1.0])
    elif in_dst[qi]:
      expected = inv @ np.array([pts[qi, 0], pts[qi, 1], 1.0])
    else:
      continue
    if np.linalg.norm(expected - pts[ti]) <= args.tol_px:
      correct_by_query.add(qi)
      correct_pairs.add((min(qi, ti), max(qi, ti)))
  row["n_good_matches_correct"] = len(correct_by_query)
  row["n_correct_pairs_distinct"] = len(correct_pairs)

  # --- RANSAC clusters, exactly as run_sift_layer builds them ------------------
  kept = keep_one_direction(match_idx)
  row["n_matches_kept"] = len(kept)
  row["n_matches_kept_correct"] = sum(1 for q, t in kept if q in correct_by_query)
  t0 = time.perf_counter()
  clusters = run_RANSAC([(kp[q], kp[t]) for q, t in kept], RANSAC_MAX_CLUSTERS,
                        args.ransac_threshold, args.ransac_min_inliers,
                        image_shape=gray.shape)
  row["sec_ransac"] = round(time.perf_counter() - t0, 4)
  score_clusters(clusters, matrix, inv, src_mask, dst_mask, args.tol_px, row)
  if row["cluster_found"]:
    row["cluster_outcome"] = "found"
  elif row["n_good_matches_correct"] == 0:
    row["cluster_outcome"] = "no_correct_match"
  elif row["n_matches_kept_correct"] == 0:
    row["cluster_outcome"] = "dropped_by_one_direction"
  elif row["n_matches_kept_correct"] < args.ransac_min_inliers:
    row["cluster_outcome"] = "too_few_for_ransac"
  else:
    row["cluster_outcome"] = "ransac"

  # --- queries: every keypoint inside either region, both directions ----------
  queries = [(i, "src", matrix) for i in np.nonzero(in_src)[0]]
  queries += [(i, "dst", inv) for i in np.nonzero(in_dst)[0]]
  row["n_queries_total"] = len(queries)
  if args.max_queries and len(queries) > args.max_queries:
    # per-query cost is O(n_keypoints), so a dense image with 15k queries costs
    # ~30x a typical one. Subsample: the funnel is a set of proportions, and a
    # few hundred draws already pin those down. Seeded by image content so the
    # same image always yields the same sample.
    seed = int(hashlib.md5(gray.tobytes()[:4096]).hexdigest()[:8], 16)
    queries = random.Random(seed).sample(queries, args.max_queries)
  row["n_queries"] = len(queries)
  if not queries or des is None:
    row["lost_at"] = "no_kp_in_region"
    return row

  qidx = [q[0] for q in queries]
  # a low contrast threshold on a big image can yield tens of thousands of
  # keypoints; computing all query rows at once would be hundreds of MB, so
  # walk the queries in chunks and keep only one chunk of distances alive
  chunk, chunk_base = None, -1

  def finish(pair):
    # the simulation must reproduce the real pipeline query by query; any
    # disagreement means a stage is being mis-attributed (or k is too small)
    if (pair["pass_dist_first"] == 1) != bool(pair["found_by_pipeline"]):
      row["n_sim_mismatch"] += 1
  ranks_all, ranks_far, ratios_global, ratios_actual, ratios_fair = [], [], [], [], []

  for slot, (i, side, fwd) in enumerate(queries):
    expected = fwd @ np.array([pts[i, 0], pts[i, 1], 1.0])
    geo = np.linalg.norm(pts - expected, axis=1)

    # STAGE 1 -- does a keypoint exist where this one should have been copied?
    partners = np.nonzero(geo <= args.tol_px)[0]
    partners = partners[partners != i]

    if slot // QUERY_CHUNK != chunk_base:
      chunk_base = slot // QUERY_CHUNK
      chunk = _l2_rows(des, qidx[chunk_base * QUERY_CHUNK:(chunk_base + 1) * QUERY_CHUNK])
    ddesc = chunk[slot % QUERY_CHUNK].copy()
    ddesc[i] = np.inf                                   # the pipeline drops self
    self_dist = np.linalg.norm(pts - pts[i], axis=1)
    far = (self_dist > args.min_distance)               # survives min_distance
    far[i] = False

    pair = {
      "query_idx": int(i), "side": side,
      "query_x": round(float(pts[i, 0]), 1), "query_y": round(float(pts[i, 1]), 1),
      "query_size": round(float(sizes[i]), 2),
      "expected_x": round(float(expected[0]), 1),
      "expected_y": round(float(expected[1]), 1),
      "n_partners": len(partners), "detected": int(len(partners) > 0),
      "partner_dist_px": "", "partner_size_ratio": "",
      "rank_all": "", "rank_far": "", "d_partner": "", "d_nn1_far": "",
      "d_nn2_far": "", "d_best_nonpartner_far": "",
      "ratio_global": "", "ratio_far": "", "ratio_vs_nonpartner": "",
      "nn1_is_partner": "", "nn2_is_partner": "",
      "nn2_all_pixel_dist": "", "pass_ratio_first": "",
      "pass_dist_first": "", "found_by_pipeline": int(i in correct_by_query),
    }

    if len(partners) == 0:
      finish(pair)
      continue
    row["n_detected"] += 1

    best_p = partners[np.argmin(geo[partners])]
    pair["partner_dist_px"] = round(float(geo[best_p]), 2)
    pair["partner_size_ratio"] = round(float(sizes[best_p] / max(sizes[i], 1e-6)), 3)

    # STAGE 2 -- is the true partner the nearest neighbour in descriptor space?
    d_partner = float(ddesc[partners].min())
    pair["d_partner"] = round(d_partner, 2)
    rank_all = 1 + int(np.count_nonzero(ddesc < d_partner))
    pair["rank_all"] = rank_all
    ranks_all.append(rank_all)
    if rank_all == 1:
      row["n_rank1_all"] += 1
    if rank_all <= KNN_K:
      # knnMatch(k=KNN_K) is the whole candidate window the pipeline ever sees;
      # a partner ranked below it is invisible no matter what the filters do
      row["n_rank_all_le_knnk"] += 1

      # OLD ordering, kept for comparison: ratio-test the global top-2 and only
    # then apply min_distance, which lets a duplicate keypoint sitting on top of
    # the query become the m2 denominator and sink an otherwise clean match
    a1, a2 = _two_smallest(ddesc)
    is_partner = np.zeros(len(pts), dtype=bool)
    is_partner[partners] = True
    if a2 is not None:
      ratio_global = float(ddesc[a1]) / max(float(ddesc[a2]), 1e-9)
      pair["ratio_global"] = round(ratio_global, 4)
      pair["nn2_all_pixel_dist"] = round(float(self_dist[a2]), 2)
      passes = (bool(is_partner[a1]) and ratio_global < args.ratio_threshold
                and self_dist[a1] > args.min_distance)
      pair["pass_ratio_first"] = int(passes)
      if passes:
        row["n_pass_ratio_first"] += 1
      if is_partner[a1]:
        ratios_global.append(ratio_global)
        if not passes and ratio_global >= args.ratio_threshold:
          # NN1 was correct; classify what the denominator actually was
          if is_partner[a2]:
            row["n_blocked_by_duplicate_partner"] += 1
          elif self_dist[a2] <= args.min_distance:
            row["n_blocked_by_duplicate_denominator"] += 1

    partners_far = partners[far[partners]]
    if len(partners_far) == 0:
      # a correct partner exists but min_distance would discard it
      finish(pair)
      continue
    row["n_detected_far"] += 1

    d_partner_far = float(ddesc[partners_far].min())
    dfar = np.where(far, ddesc, np.inf)
    rank_far = 1 + int(np.count_nonzero(dfar < d_partner_far))
    pair["rank_far"] = rank_far
    ranks_far.append(rank_far)
    if rank_far == 1:
      row["n_rank1_far"] += 1

    # STAGE 3 -- ratio test, restricted to the keypoints min_distance allows
    nn1, nn2 = _two_smallest(dfar)
    pair["d_nn1_far"] = round(float(ddesc[nn1]), 2)
    pair["nn1_is_partner"] = int(bool(is_partner[nn1]))
    if nn2 is not None:
      pair["d_nn2_far"] = round(float(ddesc[nn2]), 2)
      pair["nn2_is_partner"] = int(bool(is_partner[nn2]))
      ratio_actual = float(ddesc[nn1]) / max(float(ddesc[nn2]), 1e-9)
      pair["ratio_far"] = round(ratio_actual, 4)
      if is_partner[nn1]:
        ratios_actual.append(ratio_actual)
        for t in RATIO_SWEEP:
          if ratio_actual < t:
            row[f"n_pass_far_{t:g}"] += 1
        if ratio_actual < args.ratio_threshold:
          row["n_pass_dist_first"] += 1
          pair["pass_dist_first"] = 1
        else:
          pair["pass_dist_first"] = 0
      else:
        pair["pass_dist_first"] = 0

    # what the ratio would be if duplicate keypoints on the same spot were
    # collapsed: the true partner against the best genuinely wrong neighbour
    d_non = float(np.where(is_partner, np.inf, dfar).min())
    if np.isfinite(d_non):
      pair["d_best_nonpartner_far"] = round(d_non, 2)
      ratio_fair = d_partner_far / max(d_non, 1e-9)
      pair["ratio_vs_nonpartner"] = round(ratio_fair, 4)
      ratios_fair.append(ratio_fair)

    finish(pair)

  row["sec_stats"] = round(time.perf_counter() - t_stats, 4)
  if row["n_queries"]:
    row["repeatability"] = round(row["n_detected"] / row["n_queries"], 4)
  for key, vals in (("median_rank_all", ranks_all), ("median_rank_far", ranks_far),
                    ("median_ratio_global", ratios_global),
                    ("median_ratio_far", ratios_actual),
                    ("median_ratio_vs_nonpartner", ratios_fair)):
    if vals:
      row[key] = round(float(np.median(vals)), 4)

  # --- which stage destroyed the evidence, in pipeline order ------------------
  if row["n_good_matches_correct"] > 0:
    row["lost_at"] = "none"
  elif row["n_detected"] == 0:
    row["lost_at"] = "detection"
  elif row["n_detected_far"] == 0:
    row["lost_at"] = "min_distance"
  elif row["n_rank1_far"] == 0:
    row["lost_at"] = "description"
  elif row["n_pass_dist_first"] == 0:
    row["lost_at"] = "ratio_test"
  else:
    # the full-ranking simulation says this should pass, so the only thing left
    # that can hide it is knnMatch(k=KNN_K) never returning the partner
    row["lost_at"] = "knn_k_truncation"

  return row


# ------------------------------------------------------------------------ main

def configure_threads():
  """Pin OpenCV's thread pool to the CPUs this job was actually given.

  On a shared node OpenCV may size its pool to every core on the machine while
  slurm only grants SLURM_CPUS_PER_TASK of them; the surplus threads then fight
  over the quota and can be throttled into a crawl.
  """
  before = cv.getNumThreads()
  n = os.environ.get("SLURM_CPUS_PER_TASK")
  if n is None and hasattr(os, "sched_getaffinity"):
    n = len(os.sched_getaffinity(0))
  n = int(n or os.cpu_count() or 1)
  cv.setNumThreads(n)
  affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "n/a"
  print(f"host={socket.gethostname()} opencv={cv.__version__} numpy={np.__version__} "
        f"node_cpus={os.cpu_count()} usable_cpus={affinity} "
        f"cv_threads={before}->{cv.getNumThreads()} "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}", flush=True)


def image_quality(gray, src_mask, dst_mask, args):
  """Cheap signals for blank, corrupted or badly exposed images.

  Two different failure causes, flagged separately:

  looks_blank  almost every pixel sits on one grey level. Measured as the
               fraction within +-2 levels of the most common level, because the
               standard deviation is not robust: a white Western blot with a
               1-px dark cropping line down one edge has std ~4 yet is empty.
  low_range    the image uses only a sliver of 0-255 (p99 - p1 small). The
               content is real but SIFT's contrastThreshold works on [0, 1]
               intensities, so a scene spanning 3..31 has ~1/8 of the contrast
               it expects and most structure falls under the threshold.
  """
  hist = np.bincount(gray.ravel(), minlength=256)
  mode = int(hist.argmax())
  near_mode = hist[max(mode - 2, 0):mode + 3].sum() / gray.size
  cdf = np.cumsum(hist) / gray.size
  p1, p99 = int(np.searchsorted(cdf, 0.01)), int(np.searchsorted(cdf, 0.99))
  return {
    "img_std": round(float(gray.std()), 3),
    "img_n_levels": int(np.count_nonzero(hist)),
    "img_min": int(gray.min()), "img_max": int(gray.max()),
    "img_p1": p1, "img_p99": p99, "img_range": p99 - p1,
    "img_mode": mode, "img_near_mode_frac": round(float(near_mode), 5),
    "looks_blank": int(near_mode >= args.blank_frac),
    "low_range": int(p99 - p1 < args.low_range),
    "src_region_std": round(float(gray[src_mask].std()), 3) if src_mask.any() else "",
    "dst_region_std": round(float(gray[dst_mask].std()), 3) if dst_mask.any() else "",
  }


def source_of(name):
  """Source dataset, which the file names carry as their first '__' field."""
  return name.split("__", 1)[0]


def copy_failure(args, reason, name):
  """Copy an image and its mask into <failed_dir>/<reason>/{images,masks}/.

  The copies are only a convenience for inspection, so a write error (a full
  quota, a flaky network filesystem) must not kill a multi-hour run: the first
  OSError is reported, copying is switched off for the rest of the run, and the
  diagnosis itself carries on.
  """
  if not args.failed_dir:
    return
  try:
    for folder, sub in ((args.image_folder, "images"), (args.mask_folder, "masks")):
      src = os.path.join(folder, name)
      if not os.path.exists(src):
        continue                    # e.g. the mask file itself is missing
      dst_dir = os.path.join(args.failed_dir, reason, sub)
      os.makedirs(dst_dir, exist_ok=True)
      shutil.copy2(src, os.path.join(dst_dir, name))
  except OSError as e:
    args.copy_error = f"{name}: {type(e).__name__}: {e}"
    args.failed_dir = None
    print(f"  !! copying failed images stopped -- {args.copy_error}\n"
          f"     the diagnosis continues; check free space/quota on the output "
          f"filesystem", flush=True)


class _CsvStream:
  """Append rows as they are produced so a killed or timed-out job keeps its
  results and `wc -l` on the file shows live progress."""

  def __init__(self, path):
    self.path, self.f, self.w = path, None, None

  def write(self, row):
    if self.f is None:
      self.f = open(self.path, "w", newline="", encoding="utf-8")
      self.w = csv.DictWriter(self.f, fieldnames=list(row.keys()))
      self.w.writeheader()
    self.w.writerow(row)

  def flush(self):
    if self.f:
      self.f.flush()

  def close(self):
    # the cluster filesystem has returned EIO here after every row was written;
    # warn instead of raising, so the summary (built from memory) still prints
    if self.f:
      try:
        self.f.close()
      except OSError as e:
        print(f"!! closing {self.path} failed ({e}); its last rows may be missing",
              file=sys.stderr, flush=True)


def run(args):
  configure_threads()
  t_list = time.perf_counter()
  names = sorted(n for n in os.listdir(args.image_folder)
                 if n.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg")))
  if args.sample and args.sample < len(names):
    # filenames are prefixed by source dataset, so the alphabetical prefix that
    # --limit takes is one dataset, not a cross-section of the split
    names = sorted(random.Random(args.seed).sample(names, args.sample))
  if args.limit:
    names = names[:args.limit]
  os.makedirs(args.out_dir, exist_ok=True)
  tag = f"_{args.split}" if args.split else ""
  img_csv = os.path.join(args.out_dir, f"sift_diagnostic{tag}_per_image.csv")

  skip_csv = os.path.join(args.out_dir, f"sift_diagnostic{tag}_skipped.csv")
  if args.copy_failed and not args.failed_dir:
    args.failed_dir = os.path.join(args.out_dir, f"failed{tag}")
  if not args.copy_failed:
    args.failed_dir = None
  args.copy_error = None

  img_rows, skipped = [], []
  img_out = _CsvStream(img_csv)
  skip_out = _CsvStream(skip_csv)

  def skip(name, reason, detail=""):
    skipped.append((name, reason))
    skip_out.write({"image_name": name, "source": source_of(name),
                    "reason": reason, "detail": detail})
    skip_out.flush()
    copy_failure(args, f"skipped_{reason}", name)
  print(f"images: {args.image_folder}\nmasks:  {args.mask_folder}\n"
        f"{len(names)} image(s) to diagnose (listing took "
        f"{time.perf_counter() - t_list:.1f}s)", flush=True)

  t_start = last_beat = time.perf_counter()
  stage_tot = {"read": 0.0, "gt": 0.0, "sift": 0.0, "match": 0.0, "ransac": 0.0,
               "stats": 0.0}

  try:
    for n_done, name in enumerate(names, 1):
      t_img = time.perf_counter()
      img = cv.imread(os.path.join(args.image_folder, name))
      mask = cv.imread(os.path.join(args.mask_folder, name))
      sec_read = time.perf_counter() - t_img
      if img is None or mask is None:
        mask_path = os.path.join(args.mask_folder, name)
        if img is None:
          size = os.path.getsize(os.path.join(args.image_folder, name))
          skip(name, "unreadable_image", f"{size} bytes")
        elif not os.path.exists(mask_path):
          skip(name, "missing_mask")
        else:
          skip(name, "unreadable_mask", f"{os.path.getsize(mask_path)} bytes")
        continue
      if mask.shape[:2] != img.shape[:2]:
        mask = cv.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)

      src_mask, dst_mask = split_mask(mask)
      if not src_mask.any() or not dst_mask.any():
        skip(name, "mask_without_regions",
             f"green_px={int(src_mask.sum())} red_px={int(dst_mask.sum())}")
        continue

      t0 = time.perf_counter()
      gt = estimate_gt_transform(src_mask, dst_mask)
      sec_gt = time.perf_counter() - t0
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      # quality flags describe the file as stored; SIFT sees the preprocessed
      # image, exactly as run_sift_layer does
      row = diagnose_image(preprocess(gray, args.preprocess), src_mask, dst_mask, gt, args)
      sec_total = time.perf_counter() - t_img
      quality = image_quality(gray, src_mask, dst_mask, args)
      row = {"image_name": name, "source": source_of(name),
             "height": img.shape[0], "width": img.shape[1],
             "gt_ok": int(gt[2] >= args.min_gt_iou),
             "preprocess": args.preprocess,
             **quality,
             **row,
             "sec_read": round(sec_read, 4), "sec_gt": round(sec_gt, 4),
             "sec_total": round(sec_total, 4)}
      img_rows.append(row)
      img_out.write(row)
      if row["lost_at"] != "none":
        copy_failure(args, row["lost_at"], name)
      img_out.flush()

      stage_tot["read"] += sec_read
      stage_tot["gt"] += sec_gt
      for k in ("sift", "match", "ransac", "stats"):
        stage_tot[k] += row[f"sec_{k}"]

      if sec_total >= args.slow_sec:
        print(f"  SLOW {sec_total:6.1f}s  {name}  {img.shape[1]}x{img.shape[0]}  "
              f"kp={row['n_kp_total']}  read={sec_read:.1f} gt={sec_gt:.1f} "
              f"sift={row['sec_sift']:.1f} match={row['sec_match']:.1f} "
              f"ransac={row['sec_ransac']:.1f} stats={row['sec_stats']:.1f}", flush=True)

      now = time.perf_counter()
      # heartbeat on a count OR on a clock, so a slow run is never silent;
      # image 1 always reports so the first timing shows up within seconds
      if n_done == 1 or (args.progress and n_done % args.progress == 0) \
         or now - last_beat >= args.heartbeat_sec:
        last_beat = now
        el = now - t_start
        rate = n_done / el if el > 0 else 0.0
        eta = (len(names) - n_done) / rate if rate > 0 else float("inf")
        n_ok = len(img_rows)
        avg = " ".join(f"{k}={1000 * v / max(n_ok, 1):.0f}ms" for k, v in stage_tot.items())
        print(f"  ... {n_done}/{len(names)}  {el / 60:5.1f} min  "
              f"{1000 * el / n_done:6.0f} ms/img  eta {eta / 60:5.1f} min  | avg {avg}",
              flush=True)
  finally:
    img_out.close()
    skip_out.close()

  summarise(img_rows, skipped, args, img_csv)
  return img_rows


def summarise(img_rows, skipped, args, img_csv):
  n = len(img_rows)
  print(f"\nparams: ratio={args.ratio_threshold} min_distance={args.min_distance} "
        f"contrast={args.contrast_thresh} edge={args.edge_thresh} tol_px={args.tol_px} "
        f"preprocess={args.preprocess} ransac_thr={args.ransac_threshold:g} "
        f"min_inliers={args.ransac_min_inliers}")
  print(f"images diagnosed: {n}" + (f"  (skipped {len(skipped)})" if skipped else ""))
  for why, c in collections.Counter(w for _, w in skipped).most_common():
    print(f"  skipped {c:5d}  {why}")
  if not n:
    return

  bad_gt = [r for r in img_rows if not r["gt_ok"]]
  if bad_gt:
    print(f"\n!! {len(bad_gt)} image(s) with unreliable ground-truth transform "
          f"(mask fit < {args.min_gt_iou}) -- treat their rows with suspicion")

  tot = lambda k: sum(r[k] for r in img_rows)

  def wtot(key, rows=None):
    """Query-derived counter, scaled back up when --max-queries subsampled.

    Each image's sampled count times n_queries_total/n_queries is an unbiased
    estimate of its full count. Without this, pooling the raw samples would
    silently reweight the funnel toward sparse images, which have lower
    detection rates than dense ones.
    """
    out = 0.0
    for r in (img_rows if rows is None else rows):
      used = r.get("n_queries") or 0
      total = r.get("n_queries_total") or used
      out += r[key] * (total / used if used else 0)
    return int(round(out))
  found = sum(1 for r in img_rows if r["n_good_matches_correct"] > 0)
  print(f"\nend-to-end: {found}/{n} images ({100.0*found/n:.0f}%) yielded at least "
        f"one geometrically correct match")

  cfound = [r for r in img_rows if r["cluster_found"]]
  print(f"\ncluster level (RANSAC, one direction per pair): {len(cfound)}/{n} images "
        f"({100.0 * len(cfound) / n:.0f}%) have a correct cluster")
  outcomes = collections.Counter(r["cluster_outcome"] for r in img_rows)
  for k in ("found", "no_correct_match", "dropped_by_one_direction",
            "too_few_for_ransac", "ransac"):
    if outcomes[k]:
      print(f"  {k:26s} {outcomes[k]:5d}  ({100.0 * outcomes[k] / n:.0f}%)")
  n_false = tot("n_clusters_false")
  print(f"  false clusters: {n_false} in total, {n_false / n:.2f} per image; "
        f"{sum(1 for r in img_rows if r['n_clusters_false'] == 0)}/{n} images have none")
  if cfound:
    ranks = [r["first_correct_rank"] for r in cfound]
    print(f"  first correct cluster is RANSAC's #1 in "
          f"{100.0 * sum(1 for x in ranks if x == 1) / len(ranks):.0f}% of found images, "
          f"top-3 in {100.0 * sum(1 for x in ranks if x <= 3) / len(ranks):.0f}%")
    accurate = sum(1 for r in cfound if r["best_cluster_transform_err"] <= args.tol_px)
    print(f"  best correct cluster's matrix within {args.tol_px:g}px of the truth over the "
          f"whole region: "
          f"{accurate}/{len(cfound)} ({100.0 * accurate / len(cfound):.0f}%)")
    print(f"  best correct cluster vs the true regions: median coverage "
          f"{np.median([r['best_cluster_coverage'] for r in cfound]):.2f}, median purity "
          f"{np.median([r['best_cluster_purity'] for r in cfound]):.2f}")
  mirrored = tot("cluster_mirrored")
  if mirrored:
    print(f"  !! {mirrored} image(s) still have the forgery as two mirrored clusters")

  print("\nwhere the correct pairs die (images):")
  order = ["none", "no_keypoints", "no_kp_in_region", "detection", "min_distance",
           "description", "ratio_test", "knn_k_truncation"]
  counts = {k: sum(1 for r in img_rows if r["lost_at"] == k) for k in order}
  for k in order:
    if counts[k]:
      print(f"  {k:26s} {counts[k]:4d}  ({100.0*counts[k]/n:.0f}%)")


  # ---- per source dataset ----
  by_src = collections.defaultdict(list)
  for r in img_rows:
    by_src[r["source"]].append(r)
  skip_src = collections.Counter(source_of(nm) for nm, _ in skipped)
  sources = sorted(set(by_src) | set(skip_src),
                   key=lambda k: -(len(by_src[k]) + skip_src[k]))
  print("\nper source dataset:")
  print(f"  {'source':32s} {'imgs':>5} {'skip':>4} {'found':>11} {'blank':>5} {'lowR':>4} "
        f"{'kp_med':>6} | lost: {'det':>4} {'desc':>4} {'ratio':>5} {'other':>5} | "
        f"{'det%':>5} {'NN1%':>5} {'pass%':>5} | {'>=4 pairs':>9} {'prec%':>5} | "
        f"{'clust%':>6} {'false/img':>9}")
  for k in sources:
    v = by_src[k]
    nd = len(v)
    found_k = sum(1 for r in v if r["lost_at"] == "none")
    lost = collections.Counter(r["lost_at"] for r in v)
    other = nd - found_k - lost["detection"] - lost["description"] - lost["ratio_test"]
    q = max(wtot("n_queries_total", v), 1)
    kp_med = int(np.median([r["n_kp_total"] for r in v])) if v else 0
    pct = f"{100.0 * found_k / nd:3.0f}%" if nd else "  - "
    print(f"  {k[:32]:32s} {nd + skip_src[k]:5d} {skip_src[k]:4d} "
          f"{found_k:5d} {pct} {sum(r['looks_blank'] for r in v):5d} "
          f"{sum(r['low_range'] for r in v):4d} {kp_med:6d} |"
          f"       {lost['detection']:4d} {lost['description']:4d} "
          f"{lost['ratio_test']:5d} {other:5d} | "
          f"{100.0 * wtot('n_detected', v) / q:5.1f} "
          f"{100.0 * wtot('n_rank1_far', v) / q:5.1f} "
          f"{100.0 * wtot('n_pass_dist_first', v) / q:5.1f} | "
          f"{sum(1 for r in v if r['n_correct_pairs_distinct'] >= 4):4d}/{nd:<4d} "
          f"{100.0 * sum(r['n_good_matches_correct'] for r in v) / max(sum(r['n_good_matches'] for r in v), 1):5.1f} | "
          f"{100.0 * sum(r['cluster_found'] for r in v) / max(nd, 1):5.1f}% "
          f"{sum(r['n_clusters_false'] for r in v) / max(nd, 1):9.2f}")
  print(f"  ('>=4 pairs' = images with >=4 distinct correct matches, a proxy for "
        f"RANSAC being able\n   to form a cluster; 'blank' = >={100 * args.blank_frac:g}% "
        f"of pixels within +-2 levels of one value;\n   'lowR' = p99-p1 span < "
        f"{args.low_range} grey levels;\n   'prec%' = share of all accepted matches "
        f"that are geometrically correct -- watch it\n   when comparing --preprocess "
        f"modes, which can add keypoints and false matches alike)")

  blank = [r for r in img_rows if r["looks_blank"]]
  if blank:
    found_blank = sum(1 for r in blank if r["lost_at"] == "none")
    print(f"\n!! {len(blank)} near-blank image(s): nothing to match, excluding them is "
          f"legitimate." + (f"\n   {found_blank} of them were FOUND anyway -- that "
          f"contradicts 'blank', so check\n   --blank-frac before trusting this flag."
          if found_blank else ""))
  low = [r for r in img_rows if r["low_range"] and not r["looks_blank"]]
  if low:
    fl = sum(1 for r in low if r["lost_at"] == "none")
    fo = sum(1 for r in img_rows if not r["low_range"] and r["lost_at"] == "none")
    no = sum(1 for r in img_rows if not r["low_range"])
    print(f"\n!! {len(low)} low-range image(s) (real content squeezed into < "
          f"{args.low_range} grey levels):\n   found {100.0 * fl / len(low):.0f}% vs "
          f"{100.0 * fo / max(no, 1):.0f}% for the rest. These are exposure problems, "
          f"not corruption.")
  if args.copy_error:
    print(f"\n!! copying of failed images was stopped part-way by a write error:"
          f"\n   {args.copy_error}\n   the CSV is complete; the failed_* folder "
          f"is not.")
  elif args.failed_dir and (skipped or found < n):
    print(f"\nfailing and skipped images copied to {args.failed_dir}/<reason>/"
          f"{{images,masks}}")

  n_tot = sum(r.get("n_queries_total", 0) or 0 for r in img_rows)
  n_used = tot("n_queries")
  if n_tot > n_used:
    print(f"\n(--max-queries={args.max_queries}: funnel measured on {n_used} of "
          f"{n_tot} correspondences ({100.0 * n_used / n_tot:.0f}%) and scaled back "
          f"up per image;\n funnel numbers are estimates, end-to-end counts are exact)")
  print("\nstage funnel (ground-truth correspondences, both directions):")
  stages = [
    ("keypoints inside a region  ", "n_queries_total"),
    ("1 detection: partner exists", "n_detected"),
    ("  survives min_distance    ", "n_detected_far"),
    ("2 description: partner NN1 ", "n_rank1_far"),
    ("  NN1 before any filtering ", "n_rank1_all"),
    (f"  inside knnMatch k={KNN_K}", "n_rank_all_le_knnk"),
    ("3 ratio test passes        ", "n_pass_dist_first"),
    ("  (old ratio-first order)  ", "n_pass_ratio_first"),
  ]
  base = max(wtot("n_queries"), 1)
  for label, key in stages:
    v = wtot(key)
    print(f"  {label:<27} {v:6d}  {100.0*v/base:5.1f}% of queries")
  print(f"  (stage 1 depends on --tol-px={args.tol_px}: a looser tolerance counts "
        f"sloppier\n   localisations as detected, but those rarely survive stage 2)")

  print("\nratio-threshold sweep (correspondences passing, NN1 correct):")
  for t in RATIO_SWEEP:
    v = wtot(f"n_pass_far_{t:g}")
    print(f"  ratio < {t:<5g} {v:6d}  {100.0*v/base:5.1f}% of queries")

  gain = wtot("n_pass_dist_first") - wtot("n_pass_ratio_first")
  print(f"\nfilter order: the pipeline drops candidates within min_distance BEFORE "
        f"the\n  ratio test. Testing the ratio first instead would "
        f"{'lose' if gain >= 0 else 'gain'} {abs(gain)} correspondence(s),")
  print(f"  because a duplicate keypoint on the query would act as the m2 denominator:")
  print(f"    denominator would be a keypoint within min_distance  "
        f"{wtot('n_blocked_by_duplicate_denominator'):5d}")
  print(f"    denominator would be a second correct partner        "
        f"{wtot('n_blocked_by_duplicate_partner'):5d}")
  print(f"    NN1 status itself lost to a near-duplicate           "
        f"{wtot('n_rank1_far') - wtot('n_rank1_all'):5d}")

  # the simulation ranks ALL neighbours; the pipeline only sees k=KNN_K of them
  trunc = sum(1 for r in img_rows if r["lost_at"] == "knn_k_truncation")
  mismatch = tot("n_sim_mismatch")
  if trunc or mismatch:
    print(f"\n!! simulation and pipeline DISAGREE on {mismatch} correspondence(s) "
          f"({trunc} image(s) lost\n   to knnMatch(k={KNN_K}) truncation). Stage "
          f"attribution is unreliable for those; raise\n   KNN_K in sift_layer.py "
          f"if truncation is the cause.")
  else:
    print(f"\nsimulation and pipeline agree on every examined correspondence, so the "
          f"stage\n  attribution above is exact (knnMatch k={KNN_K} window is sufficient).")

  secs = [float(r["sec_total"]) for r in img_rows if r.get("sec_total")]
  if secs:
    tot_s = sum(secs)
    print(f"\ntiming: {tot_s / 60:.1f} min total, {1000 * tot_s / len(secs):.0f} ms/image "
          f"(median {1000 * float(np.median(secs)):.0f} ms)")
    for k in ("read", "gt", "sift", "match", "ransac", "stats"):
      v = sum(float(r.get(f"sec_{k}") or 0) for r in img_rows)
      print(f"  {k:6s} {v / 60:6.1f} min  {100.0 * v / tot_s:4.0f}%")
    worst = sorted(img_rows, key=lambda r: -float(r["sec_total"]))[:5]
    print("  slowest images:")
    for r in worst:
      print(f"    {float(r['sec_total']):6.1f}s  kp={r['n_kp_total']:>6}  "
            f"{r['width']}x{r['height']}  {r['image_name'][:54]}")

  print(f"\nwrote {img_csv}")


def parse_args():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--split", choices=("train", "test", "val"),
                 help="take the image/mask folders from config/paths.txt "
                      "(CMFD_<SPLIT>_IMAGES / _MASKS) instead of the local sample set")
  p.add_argument("--paths-file", default=os.path.join(REPO_ROOT, "config/paths.txt"),
                 help="KEY=VALUE file holding the dataset locations")
  p.add_argument("--image-folder", help="explicit image folder (overrides --split)")
  p.add_argument("--mask-folder", help="explicit mask folder (overrides --split)")
  p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "logs"))
  p.add_argument("--limit", type=int, default=0,
                 help="only the first N images ALPHABETICALLY -- filenames start "
                      "with the source dataset, so this picks one dataset, not a "
                      "cross-section. Use --sample for a representative subset.")
  p.add_argument("--sample", type=int, default=0,
                 help="randomly sample N images across the whole split (seeded)")
  p.add_argument("--seed", type=int, default=0, help="seed for --sample")
  p.add_argument("--max-queries", type=int, default=0,
                 help="cap ground-truth correspondences examined per image "
                      "(0 = all). Bounds the cost of dense images; the per-image "
                      "CSV keeps n_queries_total next to the sampled n_queries.")
  p.add_argument("--progress", type=int, default=0,
                 help="print a progress line every N images (0 = only on the clock)")
  p.add_argument("--heartbeat-sec", type=float, default=60.0,
                 help="also print a progress line at least this often, in seconds")
  p.add_argument("--no-copy-failed", dest="copy_failed", action="store_false",
                 help="do not copy failing/skipped images and masks aside")
  p.add_argument("--failed-dir",
                 help="where failing images go, one subfolder per reason "
                      "(default: <out-dir>/failed_<split>)")
  p.add_argument("--blank-frac", type=float, default=0.995,
                 help="flag looks_blank when at least this fraction of pixels is "
                      "within +-2 grey levels of the most common one")
  p.add_argument("--low-range", type=int, default=40,
                 help="flag low_range when the image's p99 - p1 grey-level span "
                      "is below this (dark or washed-out exposure)")
  p.add_argument("--slow-sec", type=float, default=10.0,
                 help="report any single image slower than this, with a per-stage breakdown")

  p.add_argument("--ratio-threshold", type=float, default=0.7)
  p.add_argument("--min-distance", type=float, default=8.0)
  p.add_argument("--contrast-thresh", type=float, default=0.01)
  p.add_argument("--edge-thresh", type=float, default=30.0)
  p.add_argument("--ransac-threshold", type=float, default=RANSAC_THRESHOLD,
                 help="RANSAC reprojection threshold in px (pipeline default "
                      f"{RANSAC_THRESHOLD:g}; copies shift exactly, so true pairs fit ~0)")
  p.add_argument("--ransac-min-inliers", type=int, default=RANSAC_MIN_INLIERS,
                 help=f"pairs needed to accept a cluster (pipeline default {RANSAC_MIN_INLIERS})")
  p.add_argument("--preprocess", choices=PREPROCESS_MODES, default=DEFAULT_PREPROCESS,
                 help="contrast normalisation before SIFT, shared with sift_layer.py "
                      f"(default {DEFAULT_PREPROCESS})")

  p.add_argument("--tol-px", type=float, default=3.0,
                 help="how far a keypoint may sit from its predicted position and "
                      "still count as the correct partner")
  p.add_argument("--min-gt-iou", type=float, default=0.5,
                 help="mask fit (overlap-aware, see _gt_score) below which the recovered transform is flagged")

  args = p.parse_args()

  # --split reads the dataset location from config/paths.txt; an explicit
  # --image-folder/--mask-folder always wins; otherwise fall back to the
  # local real_samples set that lives in the repo.
  if args.split and not (args.image_folder and args.mask_folder):
    img, mask = resolve_split(args.split, args.paths_file)
    args.image_folder = args.image_folder or img
    args.mask_folder = args.mask_folder or mask
  args.image_folder = args.image_folder or os.path.join(REPO_ROOT, "data/real_samples/images")
  args.mask_folder = args.mask_folder or os.path.join(REPO_ROOT, "data/real_samples/masks")
  return args


if __name__ == "__main__":
  # `scancel --signal=USR1 <jobid>` (or `kill -USR1 <pid>`) prints every
  # thread's current Python stack to stderr without stopping the run: the way
  # to see where a silent job is stuck, even in the middle of one image.
  if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
  run(parse_args())
