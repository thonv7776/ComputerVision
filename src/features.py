"""Task 1B: SIFT keypoints and matching with Lowe's ratio test."""
import cv2


def sift_match(gray1, gray2, ratio=0.75):
    """Match SIFT features between two grayscale images.

    A match is kept only if its distance is below ratio * the distance of the second-best
    candidate, which removes ambiguous matches. Returns kp1, kp2 and the list of good cv2.DMatch.
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(des2) < 2:     # textureless image: no descriptors
        return kp1, kp2, []
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    return kp1, kp2, good


def draw_matches(img1, kp1, img2, kp2, good, n=35):
    """Draw the n best matches (smallest distance) and return a BGR image."""
    best = sorted(good, key=lambda m: m.distance)[:n]
    return cv2.drawMatches(img1, kp1, img2, kp2, best, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
