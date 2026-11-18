# Copy-move forgery detection in bioscientific images

A copy-move forgery copies a region of an image and pastes it elsewhere in the same image, for example to duplicate a band in a Western blot or a cell in a microscopy image. This repository detects such forgeries: for each image it decides whether it contains a copy-move pair and, if so, returns where the source region and its copy are, and the transform between them.

## Pipeline

1. **Preprocess:** percentile contrast stretch, so dark and washed-out images keep their keypoints.
2. **Keypoints:** SIFT (contrast threshold 0.01, edge threshold 30). LIFT is available as an alternative.
3. **Self-matching:** each keypoint against its 10 nearest descriptors in the same image; partners closer than 8 px are dropped, then a ratio test of 0.7.
4. **Clustering:** RANSAC fits a partial affine transform (2 px threshold) to the matches, cluster by cluster.
5. **Decision:** the image is flagged when its first cluster has at least 10 inliers. That cluster gives the two regions and the transform.

## Contents

| file | what it does |
|---|---|
| `src/sift_lift_layer.py` | the detection pipeline above, with SIFT or LIFT features |
| `src/lift.py` | PyTorch port of LIFT (Yi et al., ECCV 2016) using the authors' pretrained weights, downloaded on first use |
| `src/diagnose_sift.py` | evaluation on forged images against their masks: whether each forgery is found, and at which stage it is lost when it is not |
| `src/fft_layer.py`, `src/eval_fft.py` | an experimental polar FFT verifier for clusters, and an evaluation on forged and unforged images that draws every false positive for inspection |
| `util/region_from_kp.py` | turns a cluster's keypoints into a region (convex hull and bounding box) |
| `config/paths.example.txt` | template for the dataset locations (copy to `config/paths.txt`) |

## How accuracy is measured

- **Forged images** (10,619 images from 9 source datasets) come with masks marking the source region and its copy. The true transform is recovered from the masks. A cluster is correct when most of its matched pairs land within 3 px of where the true transform sends them. An image counts as **detected** when its first cluster is correct.
- **Unforged images** (2,691) measure false alarms. Matches on annotations (sample labels, rulers, scale-bar text) are acceptable to flag; a **false positive** is an image flagged on its content (cells, particles, blots or background). Every flagged unforged image was checked by eye.

## Results

| | |
|---|---|
| forged images detected | **85.8%** (test 82.7%, val 82.5%, train 87.9%) |
| unforged images flagged on content | **0.93%** (25 of 2,691) |
| unforged images flagged including annotations | 5.9% |

Detection by source dataset: Broad Bioimage, CAMA, HeLaCytoNuc and TYC 100%, Maize Cobs 99.5%, SIID 96.6%, EMPS 95.2%, MoNuSAC 75.4%, Western blots 48.2%. Missed forgeries are mostly small copied regions that produce fewer than 10 matching keypoints.

## Running

Requires Python with `opencv-python` and `numpy`; LIFT also needs `torch` and `h5py`.

```bash
python src/diagnose_sift.py                         # local sample images in data/real_samples
python src/diagnose_sift.py --split test            # a dataset split, paths from config/paths.txt
python src/diagnose_sift.py --split test --features lift
```
