"""Two hand-gesture classifiers over MediaPipe's 21 hand landmarks (x, y, z tuples, y grows downward).
`original` is the rule set from github.com/shashwater/temple-run-gesture-controller (src/gestures.py), unchanged in logic.
`improved` is rotation-invariant and requires an exact set of extended fingers."""
import math

TIPS = (8, 12, 16, 20)  # index, middle, ring, pinky tips


def original(lm):
    """Finger counts as 'up' when tip.y is above the joint two landmarks earlier. Assumes an upright hand. Thumb ignored."""
    n = sum(lm[t][1] < lm[t - 2][1] - 0.02 for t in TIPS)
    return {0: "Slide", 1: "Left", 2: "Right", 3: "Jump"}.get(n, "Unknown")


def _d(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])


def extended(lm, ratio=1.1):
    """A finger is extended when its tip is farther from the wrist than its PIP joint (works at any hand rotation)."""
    return tuple(_d(lm[t], lm[0]) > _d(lm[t - 2], lm[0]) * ratio for t in TIPS)


_SETS = {(False,) * 4: "Slide", (True, False, False, False): "Left", (True, True, False, False): "Right", (True, True, True, False): "Jump"}


def improved(lm, ratio=1.1):
    return _SETS.get(extended(lm, ratio), "Unknown")
