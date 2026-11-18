"""How well the polar FFT verifier (fft_layer.py) removes false clusters.

Runs the SIFT layer (stretch, ratio, min_distance, RANSAC as in the pipeline)
and then the verifier on two image sets:

  forged   a split with forgery masks: every cluster is labelled correct or
           false against the transform recovered from the masks (diagnose_sift)
  base     the unforged base images: every cluster is false

Each cluster gets its polar FFT similarity, so any threshold can be evaluated
afterwards without rerunning the features. Decision rule for one threshold T,
as in run_fft_layer: clusters are checked in RANSAC order and the answer is the
first one with similarity >= T; the image is flagged when there is one. The
phase correlation peak of the same two patches is saved next to it, as a
reference. The summary compares, at each T:

  unforged flagged      base images with any passing cluster (false alarms)
  forged detected       forged images whose answer is a correct cluster
  forged wrong answer   forged images whose answer is a false cluster

against the current rule (cluster #1, at least N inliers).

False positives at --fp-threshold (default: fft_layer.MIN_SIMILARITY) are drawn
for inspection: every image whose answer is a false cluster gets a picture in
<out-dir>/false_positives/<forged|base>/<source>/, named by similarity, showing
the whole image with every false cluster that passes, and both sides of the
answer enlarged. On the base images the object masks (BASE_MASKS) also label
each cluster by where its two sides sit:
  same_object        both sides on one object (texture repeating inside it)
  different_objects  two different objects that look alike (e.g. two cells)
  background         both sides off the objects (also printed labels and text)
  object_background  one side on an object, the other off

Outputs in --out-dir: fft_clusters.csv (one row per cluster), fft_images.csv.
  python src/eval_fft.py                               # local samples
  python src/eval_fft.py --split test --base           # cluster
  python src/eval_fft.py --summarise <out-dir>         # summary from saved CSVs
"""
import argparse, collections, csv, faulthandler, os, random, signal, sys, time

import numpy as np, cv2 as cv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sift_lift_layer import (DEFAULT_PREPROCESS, PREPROCESS_MODES, RANSAC_MAX_CLUSTERS,
                             RANSAC_MIN_INLIERS, RANSAC_THRESHOLD, find_good_matches,
                             make_feature_extractor, preprocess, run_RANSAC)
from diagnose_sift import (REPO_ROOT, _CsvStream, cluster_direction, configure_threads,
                           estimate_gt_transform, load_paths, resolve_split, source_of,
                           split_mask)
from fft_layer import _HANN, MIN_SIMILARITY, run_fft_layer, side_patch

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
THRESHOLDS = (0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97)
INLIER_THRESHOLDS = (4, 6, 10, 20)
MIN_OBJECT_PX = 20      # connected components of a base mask smaller than this are noise
FP_MAX_SIDE = 1400      # longest side of the whole-image view in a false-positive picture
FP_MIN_SIDE = 700       # small images are enlarged to this
FP_CROP_H = 220         # height of each enlarged side
RELATIONS = ("background", "same_object", "different_objects", "object_background")


# ---------------------------------------------------------------- inputs

def short(source):
  """Forged names carry 'Maize_Cobs_Dataset', the base metadata 'Maize Cobs Dataset'."""
  return source.replace("_", " ").split(" (")[0].split(":")[0].strip()[:26]


def read_gray(path):
  """8-bit grey image; 16-bit and multi-channel files that IMREAD_GRAYSCALE
  rejects are read unchanged and converted."""
  g = cv.imread(path, cv.IMREAD_GRAYSCALE)
  if g is not None:
    return g
  m = cv.imread(path, cv.IMREAD_UNCHANGED)
  if m is None:
    return None
  if m.ndim == 3:
    m = cv.cvtColor(m[:, :, :3], cv.COLOR_BGR2GRAY)
  if m.dtype != np.uint8:
    m = cv.normalize(m.astype(np.float32), None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
  return m


def base_sources(img_dir, cfg):
  """Image name -> source dataset from the base metadata (file_path, origin_dataset)."""
  cands = [cfg.get("BASE_METADATA"),
           os.path.join(os.path.dirname(os.path.normpath(img_dir)), "metadata.csv"),
           os.path.join(img_dir, "metadata.csv")]
  for path in cands:
    if path and os.path.isfile(path):
      with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
      if rows and {"file_path", "origin_dataset"} <= set(rows[0]):
        print(f"base metadata: {path}", flush=True)
        return {os.path.basename(r["file_path"]): r["origin_dataset"].strip() for r in rows}
  print("!! no base metadata found; base sources left blank", flush=True)
  return {}


def find_base_mask(mask_dir, name):
  """Base masks are named <stem>_gt.png; also accept the image's own name."""
  if not mask_dir:
    return None
  stem = os.path.splitext(name)[0]
  for cand in (f"{stem}_gt.png", name, f"{stem}.png"):
    path = os.path.join(mask_dir, cand)
    if os.path.isfile(path):
      return path
  return None


def object_labels(mask_path, shape):
  """Connected objects of a binary base mask (0 = background); None without one."""
  m = cv.imread(mask_path, cv.IMREAD_UNCHANGED) if mask_path else None
  if m is None:
    return None
  if m.ndim == 3:
    m = m[:, :, :3].max(axis=2)
  if m.shape[:2] != shape:
    m = cv.resize(m, (shape[1], shape[0]), interpolation=cv.INTER_NEAREST)
  n, labels, stats, _ = cv.connectedComponentsWithStats((m > 127).astype(np.uint8), connectivity=8)
  small = [i for i in range(1, n) if stats[i, cv.CC_STAT_AREA] < MIN_OBJECT_PX]
  if small:
    labels[np.isin(labels, small)] = 0
  return labels


def relation(labels, a, b):
  """Where a cluster's two sides sit on the object mask (see the module doc)."""
  if labels is None:
    return ""
  def side(pts):
    h, w = labels.shape
    lab = labels[np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1),
                 np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)]
    on = lab > 0
    return on.mean() >= 0.5, (np.bincount(lab[on]).argmax() if on.any() else 0)
  (on_a, obj_a), (on_b, obj_b) = side(a), side(b)
  if on_a and on_b:
    return "same_object" if obj_a == obj_b else "different_objects"
  return "background" if not (on_a or on_b) else "object_background"


def list_images(folder, args):
  names = sorted(n for n in os.listdir(folder) if n.lower().endswith(IMAGE_EXTS))
  if args.sample and args.sample < len(names):
    names = sorted(random.Random(args.seed).sample(names, args.sample))
  if args.limit:
    names = names[:args.limit]
  return names


# ---------------------------------------------------------------- per image

def clusters_of(gray, extract, args):
  kp, des = extract(preprocess(gray, args.preprocess))
  m = find_good_matches(kp, des, args.ratio_threshold, args.min_distance)
  clusters = run_RANSAC([(kp[a], kp[b]) for a, b in m], RANSAC_MAX_CLUSTERS,
                        args.ransac_threshold, args.ransac_min_inliers)
  return len(kp), clusters


def transform_params(h):
  """Partial affine [[s cos, -s sin, tx], [s sin, s cos, ty]] -> scale, rotation (deg)."""
  return float(np.hypot(h[0, 0], h[1, 0])), float(np.degrees(np.arctan2(h[1, 0], h[0, 0])))


def draw_false_positive(pre, clusters, crow, answer, title, path):
  """Whole image with every passing false cluster (green = source side hull,
  red = copy side hull, yellow = inlier pairs of the answer), and both sides of
  the answer enlarged underneath."""
  h, w = pre.shape
  f = min(FP_MAX_SIDE / max(h, w), max(1.0, FP_MIN_SIDE / max(h, w)))
  view = cv.cvtColor(cv.resize(pre, (max(1, round(w * f)), max(1, round(h * f))),
                               interpolation=cv.INTER_AREA if f < 1 else cv.INTER_NEAREST),
                     cv.COLOR_GRAY2BGR)
  for i, r in enumerate(crow):
    if r["correct"] or r["fft_similarity"] < title["threshold"]:
      continue
    c = clusters[i]
    th = 2 if i == answer else 1
    if i == answer:
      for p, q in zip(*c["kp_pairs"]):
        cv.line(view, tuple(np.int32(p * f)), tuple(np.int32(q * f)), (0, 220, 255), 1, cv.LINE_AA)
    for key, col in (("src_region", (0, 200, 0)), ("dst_region", (0, 0, 230))):
      cv.polylines(view, [np.int32(c[key]["hull"] * f)], True, col, th, cv.LINE_AA)
    x, y = np.int32(c["src_region"]["hull"].reshape(-1, 2).min(axis=0) * f)
    cv.putText(view, f"#{i + 1}", (int(x), max(12, int(y) - 4)), cv.FONT_HERSHEY_SIMPLEX,
               0.5, (0, 200, 0), 1, cv.LINE_AA)
  crops = []
  for key, col in (("src_region", (0, 200, 0)), ("dst_region", (0, 0, 230))):
    x, y, bw, bh = clusters[answer][key]["bbox"]
    cw, ch = int(max(bw, 16) * 1.5), int(max(bh, 16) * 1.5)
    crop = cv.getRectSubPix(pre, (cw, ch), (x + bw / 2.0, y + bh / 2.0))
    sc = min(FP_CROP_H / ch, 600 / cw)
    crop = cv.cvtColor(cv.resize(crop, (max(1, round(cw * sc)), max(1, round(ch * sc))),
                                 interpolation=cv.INTER_NEAREST if sc > 1 else cv.INTER_AREA),
                       cv.COLOR_GRAY2BGR)
    cv.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), col, 2)
    crops.append(crop)
  gap = np.full((max(c.shape[0] for c in crops), 12, 3), 255, np.uint8)
  pad = lambda im, hh: np.vstack([im, np.full((hh - im.shape[0], im.shape[1], 3), 255, np.uint8)])
  strip = np.hstack([pad(crops[0], gap.shape[0]), gap, pad(crops[1], gap.shape[0])])
  text_w = max(cv.getTextSize(l, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for l in title["lines"])
  width = max(view.shape[1], strip.shape[1], text_w + 16)
  widen = lambda im: np.hstack([im, np.full((im.shape[0], width - im.shape[1], 3), 255, np.uint8)])
  band = np.full((24 * len(title["lines"]) + 10, width, 3), 255, np.uint8)
  for j, line in enumerate(title["lines"]):
    cv.putText(band, line, (8, 24 * (j + 1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
  sep = np.full((10, width, 3), 255, np.uint8)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  cv.imwrite(path, np.vstack([band, widen(view), sep, widen(strip)]), [cv.IMWRITE_JPEG_QUALITY, 90])


def save_false_positive(kind, name, source, pre, clusters, crow, args):
  """Picture of the image when its answer at --fp-threshold is a false cluster.
  Returns the file written ("" when the answer is correct or there is none).
  A write error stops the pictures for the rest of the run, not the run."""
  t = args.fp_threshold
  answer = next((i for i, r in enumerate(crow) if r["fft_similarity"] >= t), None)
  if not args.fp_dir or answer is None or crow[answer]["correct"]:
    return ""
  a = crow[answer]
  n_false = sum(not r["correct"] and r["fft_similarity"] >= t for r in crow)
  lines = [f"{kind} | {source or '(no source)'} | {name}",
           f"answer: cluster #{a['rank']} of {len(crow)}, {a['inliers']} inliers, scale "
           f"{a['scale']:.3f}, rot {a['rot_deg']:.1f} deg" +
           (f", {a['relation']}" if a.get("relation") else ""),
           f"polar FFT {a['fft_similarity']:.3f}, phase corr {a['phase_corr']:.3f} | "
           f"{n_false} false cluster(s) pass polar FFT >= {t:g}",
           "green = source side, red = copy side, yellow = inlier pairs of the answer; "
           "below: both sides of the answer"]
  folder = source.replace(" ", "_") or "no_source"
  path = os.path.join(args.fp_dir, kind, folder,
                      f"{a['fft_similarity']:.3f}_{os.path.splitext(name)[0]}.jpg")
  try:
    draw_false_positive(pre, clusters, crow, answer, {"threshold": t, "lines": lines}, path)
  except (OSError, cv.error) as e:
    print(f"  !! false-positive pictures stopped -- {name}: {type(e).__name__}: {e}", flush=True)
    args.fp_dir = None
    return ""
  return os.path.relpath(path, args.out_dir)


def run_image(kind, name, source, gray, masks, extract, args, labels=None):
  t0 = time.perf_counter()
  n_kp, clusters = clusters_of(gray, extract, args)
  t1 = time.perf_counter()
  pre = preprocess(gray, args.preprocess)
  run_fft_layer(pre, clusters)
  for c in clusters:
    a, b = (side_patch(pre, c[k]).astype(np.float64) for k in ("src_region", "dst_region"))
    c["phase_corr"] = cv.phaseCorrelate(a, b, _HANN.astype(np.float64))[1]
  t2 = time.perf_counter()
  if masks is not None:
    matrix, _, _ = estimate_gt_transform(*masks)
    inv = cv.invertAffineTransform(matrix)
  crow = []
  for rank, c in enumerate(clusters, 1):
    correct = 0
    if masks is not None:
      correct = int(max(cluster_direction(c, matrix, inv, args.tol_px)) >= 0.5)
    scale, rot = transform_params(c["h_matrix"])
    crow.append({"kind": kind, "image_name": name, "source": source, "rank": rank,
                 "inliers": c["inlier_count"], "correct": correct,
                 "fft_similarity": round(c["fft_similarity"], 4),
                 "phase_corr": round(c["phase_corr"], 4),
                 "scale": round(scale, 4), "rot_deg": round(rot, 2),
                 "tx": round(float(c["h_matrix"][0, 2]), 1), "ty": round(float(c["h_matrix"][1, 2]), 1),
                 "shift_px": round(float(np.hypot(*c["h_matrix"][:, 2])), 1),
                 "relation": relation(labels, *c["kp_pairs"]),
                 "src_box": "x".join(map(str, c["src_region"]["bbox"][2:])),
                 "dst_box": "x".join(map(str, c["dst_region"]["bbox"][2:]))})
  fp_file = save_false_positive(kind, name, source, pre, clusters, crow, args)
  row = {"kind": kind, "image_name": name, "source": source,
         "height": gray.shape[0], "width": gray.shape[1], "n_kp_total": n_kp,
         "n_clusters": len(clusters), "n_clusters_correct": sum(c["correct"] for c in crow),
         "has_object_mask": int(labels is not None), "fp_file": fp_file,
         "sec_layer": round(t1 - t0, 4), "sec_fft": round(t2 - t1, 4)}
  return row, crow


def run_set(kind, img_dir, mask_dir, sources, extract, args, img_out, cl_out, rows, clusters,
            object_dir=None):
  names = list_images(img_dir, args)
  print(f"\n=== {kind}: {img_dir}\n{len(names)} image(s)", flush=True)
  skipped = collections.Counter()
  t_start = last = time.perf_counter()
  for i, name in enumerate(names, 1):
    gray = read_gray(os.path.join(img_dir, name))
    masks = None
    if gray is None:
      skipped["unreadable_image"] += 1
      continue
    if mask_dir:
      mask = cv.imread(os.path.join(mask_dir, name))
      if mask is None:
        skipped["missing_mask"] += 1
        continue
      if mask.shape[:2] != gray.shape:
        mask = cv.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv.INTER_NEAREST)
      masks = split_mask(mask)
      if not masks[0].any() or not masks[1].any():
        skipped["mask_without_regions"] += 1
        continue
    source = sources.get(name) or (source_of(name) if "__" in name else "")
    labels = object_labels(find_base_mask(object_dir, name), gray.shape) if object_dir else None
    row, crow = run_image(kind, name, short(source), gray, masks, extract, args, labels)
    rows.append(row); clusters.extend(crow)
    img_out.write(row)
    for c in crow:
      cl_out.write(c)
    now = time.perf_counter()
    if i == 1 or (args.progress and i % args.progress == 0) or now - last >= args.heartbeat_sec:
      last = now
      el = now - t_start
      print(f"  ... {i}/{len(names)}  {el / 60:5.1f} min  {1000 * el / i:6.0f} ms/img  "
            f"eta {(len(names) - i) * el / i / 60:5.1f} min", flush=True)
      img_out.flush(); cl_out.flush()
  if skipped:
    print(f"!! {kind}: skipped {dict(skipped)}", flush=True)


def run(args):
  configure_threads()
  extract = make_feature_extractor("sift", args.contrast_thresh, args.edge_thresh)
  os.makedirs(args.out_dir, exist_ok=True)
  img_csv, cl_csv = (os.path.join(args.out_dir, f) for f in ("fft_images.csv", "fft_clusters.csv"))
  img_out, cl_out = _CsvStream(img_csv), _CsvStream(cl_csv)
  rows, clusters = [], []
  try:
    run_set("forged", args.image_folder, args.mask_folder, {}, extract, args,
            img_out, cl_out, rows, clusters)
    if args.base:
      cfg = load_paths(args.paths_file) if os.path.isfile(args.paths_file) else {}
      base_dir = args.base_folder or cfg.get("BASE_IMAGES")
      if not base_dir or not os.path.isdir(base_dir):
        raise SystemExit(f"base image folder {base_dir!r} is not a directory "
                         f"(set BASE_IMAGES in {args.paths_file} or pass --base-folder)")
      obj_dir = args.base_mask_folder or cfg.get("BASE_MASKS")
      if obj_dir and not os.path.isdir(obj_dir):
        print(f"!! base mask folder {obj_dir!r} is not a directory; no object labels", flush=True)
        obj_dir = None
      run_set("base", base_dir, None, base_sources(base_dir, cfg), extract, args,
              img_out, cl_out, rows, clusters, obj_dir)
  finally:
    img_out.close(); cl_out.close()
  summarise(rows, clusters)
  print(f"\nwrote {img_csv}\nwrote {cl_csv}")


# ---------------------------------------------------------------- summary

def auc(pos, neg):
  """Probability that a random positive scores above a random negative."""
  if not pos or not neg:
    return float("nan")
  s = np.concatenate([pos, neg])
  _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
  ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inv]     # average rank over ties
  return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def decide(cl, t, min_inliers=0, key="fft_similarity"):
  """Answer cluster for one image: the first, in RANSAC order, that passes."""
  return next((c for c in cl if c[key] >= t and c["inliers"] >= min_inliers), None)


def summarise(rows, clusters):
  by_img = collections.defaultdict(list)
  for c in clusters:
    by_img[(c["kind"], c["image_name"])].append(c)
  forged = [r for r in rows if r["kind"] == "forged"]
  base = [r for r in rows if r["kind"] == "base"]
  pct = lambda a, n: f"{100 * a / n:5.1f}%" if n else "   - "

  print(f"\nimages: {len(forged)} forged, {len(base)} unforged")
  for key in ("fft_similarity", "phase_corr", "inliers"):
    sim = lambda f: [c[key] for c in clusters if f(c)]
    groups = (("forged, correct", sim(lambda c: c["kind"] == "forged" and c["correct"])),
              ("forged, false", sim(lambda c: c["kind"] == "forged" and not c["correct"])),
              ("unforged (all false)", sim(lambda c: c["kind"] == "base")))
    print(f"\n{key} per cluster:{'n':>14}     p10   median   p90")
    for name, v in groups:
      if v:
        p = np.percentile(v, (10, 50, 90))
        print(f"  {name:<24} {len(v):7d}   {p[0]:7.3f} {p[1]:7.3f} {p[2]:7.3f}")
    print(f"  AUC correct vs forged false: {auc(groups[0][1], groups[1][1]):.3f}   "
          f"correct vs unforged: {auc(groups[0][1], groups[2][1]):.3f}")

  def table(f_rows, b_rows, rules):
    print(f"  {'rule':<26} {'unforged flagged':>17} {'forged detected':>16} {'forged wrong answer':>20}")
    for label, pick in rules:
      bf = sum(pick(by_img[("base", r["image_name"])]) is not None for r in b_rows)
      fd = fw = 0
      for r in f_rows:
        a = pick(by_img[("forged", r["image_name"])])
        fd += a is not None and a["correct"]
        fw += a is not None and not a["correct"]
      print(f"  {label:<26} {pct(bf, len(b_rows)):>17} {pct(fd, len(f_rows)):>16} "
            f"{pct(fw, len(f_rows)):>20}")

  top1 = lambda n: (lambda cl: cl[0] if cl and cl[0]["inliers"] >= n else None)
  rules = [(f"cluster #1, >= {n} inliers", top1(n)) for n in INLIER_THRESHOLDS]
  rules += [(f"similarity >= {t:g}", (lambda t: lambda cl: decide(cl, t))(t)) for t in THRESHOLDS]
  rules += [(f"sim >= {t:g}, >= 6 inliers", (lambda t: lambda cl: decide(cl, t, 6))(t))
            for t in THRESHOLDS]
  rules += [(f"phase corr >= {t:g}", (lambda t: lambda cl: decide(cl, t, 0, "phase_corr"))(t))
            for t in THRESHOLDS]
  print("\nall datasets")
  table(forged, base, rules)

  if any(c.get("relation") for c in clusters if c["kind"] == "base"):
    print("\nunforged images flagged, by where the answer cluster sits on the object mask")
    print(f"  {'rule':<26}" + "".join(f"{r:>19}" for r in RELATIONS))
    for label, pick in rules[:1] + rules[len(INLIER_THRESHOLDS):len(INLIER_THRESHOLDS) + len(THRESHOLDS)]:
      n = collections.Counter()
      for r in base:
        a = pick(by_img[("base", r["image_name"])])
        if a is not None:
          n[a["relation"]] += 1
      print(f"  {label:<26}" + "".join(f"{n[x]:>19}" for x in RELATIONS))

  sources = sorted({r["source"] for r in rows}, key=lambda s: -sum(r["source"] == s for r in rows))
  for s in sources:
    f_rows = [r for r in forged if r["source"] == s]
    b_rows = [r for r in base if r["source"] == s]
    print(f"\n{s or '(no source)'}: {len(f_rows)} forged, {len(b_rows)} unforged")
    table(f_rows, b_rows, rules[:1] + rules[len(INLIER_THRESHOLDS):len(INLIER_THRESHOLDS) + len(THRESHOLDS)])


def load_saved(out_dir):
  num = {"rank", "inliers", "correct", "n_clusters", "n_clusters_correct", "height", "width",
         "n_kp_total"}
  def read(name):
    with open(os.path.join(out_dir, name), encoding="utf-8", newline="") as f:
      out = []
      for r in csv.DictReader(f):
        for k in r:
          if k in num:
            r[k] = int(r[k])
          elif k in ("fft_similarity", "phase_corr"):
            r[k] = float(r[k])
        out.append(r)
      return out
  return read("fft_images.csv"), read("fft_clusters.csv")


def parse_args():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--summarise", metavar="OUT_DIR", help="only print the summary of a saved run")
  p.add_argument("--split", choices=("train", "test", "val"),
                 help="forged images from config/paths.txt instead of the local samples")
  p.add_argument("--paths-file", default=os.path.join(REPO_ROOT, "config/paths.txt"))
  p.add_argument("--image-folder"); p.add_argument("--mask-folder")
  p.add_argument("--base", action="store_true", help="also run the unforged base images")
  p.add_argument("--base-folder", help="unforged images (default BASE_IMAGES in the paths file)")
  p.add_argument("--base-mask-folder",
                 help="object masks of the unforged images (default BASE_MASKS in the paths file)")
  p.add_argument("--fp-threshold", type=float, default=MIN_SIMILARITY,
                 help="polar FFT threshold at which false positives are drawn "
                      f"(default {MIN_SIMILARITY:g}; -1 = every image whose cluster #1 is false)")
  p.add_argument("--no-fp-pictures", dest="fp_pictures", action="store_false",
                 help="do not draw the false positives")
  p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "logs", "fft_local"))
  p.add_argument("--sample", type=int, default=0, help="random N images per set (seeded)")
  p.add_argument("--limit", type=int, default=0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--progress", type=int, default=0)
  p.add_argument("--heartbeat-sec", type=float, default=60.0)
  p.add_argument("--ratio-threshold", type=float, default=0.7)
  p.add_argument("--min-distance", type=float, default=8.0)
  p.add_argument("--contrast-thresh", type=float, default=0.01)
  p.add_argument("--edge-thresh", type=float, default=30.0)
  p.add_argument("--ransac-threshold", type=float, default=RANSAC_THRESHOLD)
  p.add_argument("--ransac-min-inliers", type=int, default=RANSAC_MIN_INLIERS)
  p.add_argument("--preprocess", choices=PREPROCESS_MODES, default=DEFAULT_PREPROCESS)
  p.add_argument("--tol-px", type=float, default=3.0)
  args = p.parse_args()
  if args.split and not (args.image_folder and args.mask_folder):
    img, mask = resolve_split(args.split, args.paths_file)
    args.image_folder = args.image_folder or img
    args.mask_folder = args.mask_folder or mask
  args.image_folder = args.image_folder or os.path.join(REPO_ROOT, "data/real_samples/images")
  args.mask_folder = args.mask_folder or os.path.join(REPO_ROOT, "data/real_samples/masks")
  args.fp_dir = os.path.join(args.out_dir, "false_positives") if args.fp_pictures else None
  return args


if __name__ == "__main__":
  if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
  a = parse_args()
  if a.summarise:
    summarise(*load_saved(a.summarise))
  else:
    run(a)
