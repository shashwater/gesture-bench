import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gesture_bench.classifiers import original, improved


def hand(extended, angle=0.0):
    """Synthetic 21-landmark hand: wrist at origin, fingers along +y (image up = -y), rotated by `angle` radians."""
    lm = [(0.0, 0.0, 0.0)] * 21; xs = {8: -0.06, 12: -0.02, 16: 0.02, 20: 0.06}
    for t, ext in zip((8, 12, 16, 20), extended):
        x = xs[t]; mcp, pip, dip, tip = (0.3, 0.45, 0.58, 0.7) if ext else (0.3, 0.38, 0.30, 0.22)
        for idx, y in zip((t - 3, t - 2, t - 1, t), (mcp, pip, dip, tip)): lm[idx] = (x, -y if ext else -y, 0.0)
    c, s = math.cos(angle), math.sin(angle)
    return [(x * c - y * s + 0.5, x * s + y * c + 0.5, z) for x, y, z in lm]


CASES = {(False,) * 4: "Slide", (True, False, False, False): "Left", (True, True, False, False): "Right", (True, True, True, False): "Jump"}


def test_both_agree_when_upright():
    for ext, name in CASES.items(): assert improved(hand(ext)) == name and original(hand(ext)) == name


def test_improved_is_rotation_invariant_original_is_not():
    for ext, name in CASES.items():
        for deg in (30, 60, 90, 180, 270):
            assert improved(hand(ext, math.radians(deg))) == name, (name, deg)
    assert any(original(hand(ext, math.radians(90))) != name for ext, name in CASES.items())


def test_exact_finger_sets():
    assert improved(hand((True, False, True, False))) == "Unknown"       # index + ring is not "two fingers"
    assert original(hand((True, False, True, False))) == "Right"          # the original counts fingers, not which ones
    assert improved(hand((True,) * 4)) == "Unknown"


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
