import cv2 as cv, numpy as np


def _region(pts: np.float64):
  # cv.convexHull only accepts float32 / int32 points; run_RANSAC passes float64
  hull = cv.convexHull(np.asarray(pts, np.float32), returnPoints=True).astype(np.int32)
  x, y, w, h = cv.boundingRect(hull)
  return {"hull": hull, "bbox": (x, y, w, h)}

def region_patch(gray, region):
  x, y, w, h = region["bbox"]
  mask = np.zeros(gray.shape[:2], np.uint8)
  cv.fillConvexPoly(mask, region["hull"], 255)
  return (gray * (mask > 0))[y:y+h, x:x+w]