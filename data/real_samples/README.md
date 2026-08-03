# Real-sample tuning set

Clean (un-annotated) copy-move pairs used to tune/validate Stage 2 locally. The
image/mask pixels are **git-ignored** (large + possibly license-restricted); only
this folder structure is tracked.

```
data/real_samples/images/<name>.png   # the forged scientific image
data/real_samples/masks/<name>.png    # RGB mask: RED=forged, GREEN=source, BLUE=background
data/real_samples/base_examples/images/<name>.png # the base example images with no forgery
data/real_samples/base_examples/masks/<name>_gt.png # masks for objects in the image, but no copy-move forgery

```

- Use the **same base filename** in `images/` and `masks/` except for the base examples.
