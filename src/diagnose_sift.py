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
"""

import argparse, csv, os
import numpy as np, cv2 as cv

from sift_layer import KNN_K, find_good_matches

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


def _iou(a, b):
  union = np.count_nonzero(a | b)
  return float(np.count_nonzero(a & b)) / union if union else 0.0


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


def estimate_gt_transform(src, dst):
  """
  Tries pure translation first and only falls back to a moment-based similarity search if
  translation fails to explain the masks. Returns (matrix, method, iou).
  """
  cs, cd = _centroid(src), _centroid(dst)

  trans = np.float32([[1, 0, cd[0] - cs[0]], [0, 1, cd[1] - cs[1]]])
  best = (trans, "translation", _iou(_warp(src, trans, dst.shape), dst))

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
        score = _iou(_warp(src, cand, dst.shape), dst)
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


def diagnose_image(gray, src_mask, dst_mask, gt, args):
  matrix, method, gt_iou = gt
  inv = cv.invertAffineTransform(matrix)

  sift = cv.SIFT_create(contrastThreshold=args.contrast_thresh,
                        edgeThreshold=args.edge_thresh)
  kp, des = sift.detectAndCompute(gray, None)

  row = {
    "gt_method": method,
    "gt_mask_iou": round(gt_iou, 4),
    "gt_dx": round(float(matrix[0, 2]), 2),
    "gt_dy": round(float(matrix[1, 2]), 2),
    "src_area_px": int(np.count_nonzero(src_mask)),
    "dst_area_px": int(np.count_nonzero(dst_mask)),
    "n_kp_total": len(kp),
  }
  counters = ["n_kp_src", "n_kp_dst", "n_queries", "n_detected", "n_detected_far",
              "n_rank1_far", "n_rank1_all", "n_rank_all_le_knnk",
              "n_pass_ratio_first", "n_pass_dist_first",
              "n_blocked_by_duplicate_partner", "n_blocked_by_duplicate_denominator",
              "n_good_matches", "n_good_matches_correct", "n_correct_pairs_distinct"]
  row.update({c: 0 for c in counters})
  row.update({f"n_pass_far_{t:g}": 0 for t in RATIO_SWEEP})
  row.update({"repeatability": "", "median_rank_all": "", "median_rank_far": "",
              "median_ratio_global": "", "median_ratio_far": "",
              "median_ratio_vs_nonpartner": "", "lost_at": ""})

  if len(kp) == 0:
    row["lost_at"] = "no_keypoints"
    return row, []

  pts = np.array([k.pt for k in kp], dtype=np.float32)
  sizes = np.array([k.size for k in kp], dtype=np.float32)
  h, w = gray.shape[:2]
  ix = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
  iy = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
  in_src, in_dst = src_mask[iy, ix], dst_mask[iy, ix]
  row["n_kp_src"], row["n_kp_dst"] = int(in_src.sum()), int(in_dst.sum())

  # --- the pipeline's own output, as the end-to-end truth for this image ------
  match_idx = find_good_matches(kp, des, args.ratio_threshold, args.min_distance)
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

  # --- queries: every keypoint inside either region, both directions ----------
  queries = [(i, "src", matrix) for i in np.nonzero(in_src)[0]]
  queries += [(i, "dst", inv) for i in np.nonzero(in_dst)[0]]
  row["n_queries"] = len(queries)
  if not queries or des is None:
    row["lost_at"] = "no_kp_in_region"
    return row, []

  qidx = [q[0] for q in queries]
  # a low contrast threshold on a big image can yield tens of thousands of
  # keypoints; computing all query rows at once would be hundreds of MB, so
  # walk the queries in chunks and keep only one chunk of distances alive
  chunk, chunk_base = None, -1

  pair_rows = []
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
      pair_rows.append(pair)
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
    order_all = np.argsort(ddesc, kind="stable")
    a1 = order_all[0]
    a2 = order_all[1] if len(order_all) > 1 else None
    is_partner = np.zeros(len(pts), dtype=bool)
    is_partner[partners] = True
    if a2 is not None and np.isfinite(ddesc[a2]):
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
      pair_rows.append(pair)
      continue
    row["n_detected_far"] += 1

    d_partner_far = float(ddesc[partners_far].min())
    far_idx = np.nonzero(far)[0]
    rank_far = 1 + int(np.count_nonzero(ddesc[far_idx] < d_partner_far))
    pair["rank_far"] = rank_far
    ranks_far.append(rank_far)
    if rank_far == 1:
      row["n_rank1_far"] += 1

    # STAGE 3 -- ratio test, restricted to the keypoints min_distance allows
    order = far_idx[np.argsort(ddesc[far_idx], kind="stable")]
    nn1 = order[0]
    nn2 = order[1] if len(order) > 1 else None
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
    nonpartner_far = far_idx[~is_partner[far_idx]]
    if len(nonpartner_far):
      d_non = float(ddesc[nonpartner_far].min())
      pair["d_best_nonpartner_far"] = round(d_non, 2)
      ratio_fair = d_partner_far / max(d_non, 1e-9)
      pair["ratio_vs_nonpartner"] = round(ratio_fair, 4)
      ratios_fair.append(ratio_fair)

    pair_rows.append(pair)

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

  return row, pair_rows


# ------------------------------------------------------------------------ main

def run(args):
  names = sorted(n for n in os.listdir(args.image_folder)
                 if n.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg")))
  if args.limit:
    names = names[:args.limit]
  os.makedirs(args.out_dir, exist_ok=True)
  tag = f"_{args.split}" if args.split else ""
  img_csv = os.path.join(args.out_dir, f"sift_diagnostic{tag}_per_image.csv")
  pair_csv = os.path.join(args.out_dir, f"sift_diagnostic{tag}_per_pair.csv")

  img_rows, pair_rows_all, skipped = [], [], []
  print(f"images: {args.image_folder}\nmasks:  {args.mask_folder}\n"
        f"{len(names)} image(s) to diagnose", flush=True)

  for n_done, name in enumerate(names, 1):
    if args.progress and n_done % args.progress == 0:
      print(f"  ... {n_done}/{len(names)}", flush=True)
    img = cv.imread(os.path.join(args.image_folder, name))
    mask = cv.imread(os.path.join(args.mask_folder, name))
    if img is None or mask is None:
      skipped.append((name, "missing image or mask"))
      continue
    if mask.shape[:2] != img.shape[:2]:
      mask = cv.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)

    src_mask, dst_mask = split_mask(mask)
    if not src_mask.any() or not dst_mask.any():
      skipped.append((name, "mask has no red/green region"))
      continue

    gt = estimate_gt_transform(src_mask, dst_mask)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    row, pairs = diagnose_image(gray, src_mask, dst_mask, gt, args)
    row = {"image_name": name, "height": img.shape[0], "width": img.shape[1],
           "gt_ok": int(gt[2] >= args.min_gt_iou), **row}
    img_rows.append(row)
    if not args.no_pair_csv:
      for p in pairs:
        pair_rows_all.append({"image_name": name, **p})

  if img_rows:
    with open(img_csv, "w", newline="", encoding="utf-8") as f:
      wr = csv.DictWriter(f, fieldnames=list(img_rows[0].keys()))
      wr.writeheader()
      wr.writerows(img_rows)
  if pair_rows_all and not args.no_pair_csv:
    with open(pair_csv, "w", newline="", encoding="utf-8") as f:
      wr = csv.DictWriter(f, fieldnames=list(pair_rows_all[0].keys()))
      wr.writeheader()
      wr.writerows(pair_rows_all)

  summarise(img_rows, pair_rows_all, skipped, args, img_csv, pair_csv)
  return img_rows


def summarise(img_rows, pair_rows, skipped, args, img_csv, pair_csv):
  n = len(img_rows)
  print(f"\nparams: ratio={args.ratio_threshold} min_distance={args.min_distance} "
        f"contrast={args.contrast_thresh} edge={args.edge_thresh} tol_px={args.tol_px}")
  print(f"images diagnosed: {n}" + (f"  (skipped {len(skipped)})" if skipped else ""))
  for name, why in skipped[:10]:
    print(f"  skipped {name}: {why}")
  if len(skipped) > 10:
    print(f"  ... and {len(skipped) - 10} more skipped")
  if not n:
    return

  bad_gt = [r for r in img_rows if not r["gt_ok"]]
  if bad_gt:
    print(f"\n!! {len(bad_gt)} image(s) with unreliable ground-truth transform "
          f"(mask IoU < {args.min_gt_iou}) -- treat their rows with suspicion")

  tot = lambda k: sum(r[k] for r in img_rows)
  found = sum(1 for r in img_rows if r["n_good_matches_correct"] > 0)
  print(f"\nend-to-end: {found}/{n} images ({100.0*found/n:.0f}%) yielded at least "
        f"one geometrically correct match")

  print("\nwhere the correct pairs die (images):")
  order = ["none", "no_keypoints", "no_kp_in_region", "detection", "min_distance",
           "description", "ratio_test", "knn_k_truncation"]
  counts = {k: sum(1 for r in img_rows if r["lost_at"] == k) for k in order}
  for k in order:
    if counts[k]:
      print(f"  {k:26s} {counts[k]:4d}  ({100.0*counts[k]/n:.0f}%)")

  print("\nstage funnel (ground-truth correspondences, both directions):")
  stages = [
    ("keypoints inside a region  ", "n_queries"),
    ("1 detection: partner exists", "n_detected"),
    ("  survives min_distance    ", "n_detected_far"),
    ("2 description: partner NN1 ", "n_rank1_far"),
    ("  NN1 before any filtering ", "n_rank1_all"),
    (f"  inside knnMatch k={KNN_K}", "n_rank_all_le_knnk"),
    ("3 ratio test passes        ", "n_pass_dist_first"),
    ("  (old ratio-first order)  ", "n_pass_ratio_first"),
  ]
  base = max(tot("n_queries"), 1)
  for label, key in stages:
    v = tot(key)
    print(f"  {label:<27} {v:6d}  {100.0*v/base:5.1f}% of queries")
  print(f"  (stage 1 depends on --tol-px={args.tol_px}: a looser tolerance counts "
        f"sloppier\n   localisations as detected, but those rarely survive stage 2)")

  print("\nratio-threshold sweep (correspondences passing, NN1 correct):")
  for t in RATIO_SWEEP:
    v = tot(f"n_pass_far_{t:g}")
    print(f"  ratio < {t:<5g} {v:6d}  {100.0*v/base:5.1f}% of queries")

  gain = tot("n_pass_dist_first") - tot("n_pass_ratio_first")
  print(f"\nfilter order: the pipeline drops candidates within min_distance BEFORE "
        f"the\n  ratio test. Testing the ratio first instead would "
        f"{'lose' if gain >= 0 else 'gain'} {abs(gain)} correspondence(s),")
  print(f"  because a duplicate keypoint on the query would act as the m2 denominator:")
  print(f"    denominator would be a keypoint within min_distance  "
        f"{tot('n_blocked_by_duplicate_denominator'):5d}")
  print(f"    denominator would be a second correct partner        "
        f"{tot('n_blocked_by_duplicate_partner'):5d}")
  print(f"    NN1 status itself lost to a near-duplicate           "
        f"{tot('n_rank1_far') - tot('n_rank1_all'):5d}")

  # the simulation ranks ALL neighbours; the pipeline only sees k=KNN_K of them
  trunc = sum(1 for r in img_rows if r["lost_at"] == "knn_k_truncation")
  missed = tot("n_pass_dist_first") - tot("n_good_matches_correct")
  if trunc or missed > 0:
    print(f"\n!! knnMatch(k={KNN_K}) TRUNCATION: {missed} correspondence(s) and "
          f"{trunc} image(s) pass the\n   full-ranking simulation but the pipeline "
          f"never saw the partner. Raise KNN_K\n   in sift_layer.py.")
  else:
    print(f"\nknnMatch(k={KNN_K}) window is sufficient: every correspondence that "
          f"should pass\n  does pass in the real pipeline (simulation and pipeline "
          f"agree exactly).")

  print(f"\nwrote {img_csv}")
  if not args.no_pair_csv:
    print(f"      {pair_csv}")


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
  p.add_argument("--limit", type=int, default=0, help="only the first N images")
  p.add_argument("--no-pair-csv", action="store_true",
                 help="skip the per-correspondence CSV (it has ~13-50 rows per "
                      "image, so a 20k-image split makes a large file)")
  p.add_argument("--progress", type=int, default=0,
                 help="print a progress line every N images (0 = never)")

  p.add_argument("--ratio-threshold", type=float, default=0.7)
  p.add_argument("--min-distance", type=float, default=8.0)
  p.add_argument("--contrast-thresh", type=float, default=0.01)
  p.add_argument("--edge-thresh", type=float, default=30.0)

  p.add_argument("--tol-px", type=float, default=3.0,
                 help="how far a keypoint may sit from its predicted position and "
                      "still count as the correct partner")
  p.add_argument("--min-gt-iou", type=float, default=0.5,
                 help="mask IoU below which the recovered transform is flagged")

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
  run(parse_args())
