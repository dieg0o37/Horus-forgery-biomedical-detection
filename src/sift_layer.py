import os, csv, numpy as np, shutil, cv2 as cv

def find_good_matches(kp, descriptors, ratio_threshold, min_distance):
  if (descriptors is None) or len(descriptors) <= 2:
    return []

  bf = cv.BFMatcher()

  raw_matches = bf.knnMatch(descriptors, descriptors, k=3)

  good_matches = []
  for m in raw_matches:
    m = [matches for matches in m if matches.trainIdx != matches.queryIdx]
    if len(m) == 2:
      m1, m2 = m

      if m1.distance < ratio_threshold * m2.distance:
        # if m1.queryIdx > m1.trainIdx later if deduplication of pairs doesnt help here.

        p1 = np.array(kp[m1.queryIdx].pt)
        p2 = np.array(kp[m1.trainIdx].pt)

        pixel_dist = np.linalg.norm(p1 - p2)

        if pixel_dist > min_distance:
          good_matches.append((m1.queryIdx, m1.trainIdx))

  return good_matches


def run_RANSAC(good_matches, max_clusters, ransac_threshold, min_inliers):
  n_matches = len(good_matches)

  clusters = []
  sampling_mask = np.ones(n_matches, dtype=bool)

  for _ in range(max_clusters):
    indices = np.where(sampling_mask)[0]

    if len(indices) < min_inliers:
      break

    pts1=np.float32([good_matches[i][0].pt for i in indices])
    pts2=np.float32([good_matches[i][1].pt for i in indices])

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
    })

  return clusters






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
      edge_thresh: float
      ) -> None:

    
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

        sift = cv.SIFT_create(contrastThreshold=contrast_thresh, edgeThreshold=edge_thresh)

        kp, descriptors = sift.detectAndCompute(gray_img, None)

        if _print:
            kp_img = cv.drawKeypoints(image=gray_img, keypoints=kp, outImage=None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv.imwrite(os.path.join(image_output_folder, f'{idx}.jpg'), kp_img)

        n_kp = len(kp)

        match_idx = find_good_matches(kp, descriptors, ratio_threshold, min_distance)
        good_matches = [(kp[qi], kp[ti]) for qi, ti in match_idx]
        sus_match_count = len(good_matches)

        clusters = run_RANSAC(good_matches=good_matches, max_clusters=100, ransac_threshold=5.0, min_inliers=4)
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
      print("input format: python sift_layer.py <_print> <ratio_threshold> <min_distance> <contrast_threshold> <edge_threshold>")
      exit(1)
    ratio_threshold = float(input[1])
    min_dist = float(input[2])
    contrast_threshold = float(input[3])
    edge_threshold = float(input[4])
    run_sift_layer(
       image_folder_path=image_folder_path,
       output_csv=output_csv,
       image_output_folder=image_output_folder,
       failed_image_output_folder=failed_image_output_folder,

       _print=_print,
       ratio_threshold=ratio_threshold,
       min_distance=min_dist,
       contrast_thresh=contrast_threshold,
       edge_thresh=edge_threshold
    )
    # clusters = run_RANSAC(good_matches=good_matches, max_clusters=100, ransac_threshold=5.0, min_inliers=4)
  except TypeError:
     print("input format: python sift_layer.py <_print> <ratio_threshold> <min_distance> <contrast_threshold> <edge_threshold>")
     exit(1)
      








