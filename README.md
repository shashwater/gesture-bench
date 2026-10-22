# gesture-bench

A benchmark and an improved classifier for the hand-gesture rules in
[temple-run-gesture-controller](https://github.com/shashwater/temple-run-gesture-controller) (webcam -> OpenCV -> MediaPipe landmarks -> rules -> keyboard events).

The controller maps fist / one finger / two fingers / three fingers to slide / left / right / jump. It had no measured accuracy. This repo measures it on
public labeled images and tests a fix for its biggest weakness: the rules assume an upright hand.

## Method
- Data: random sample of 60 images per class from the public HaGRID 30k sample set (CC BY-SA 4.0), classes `fist`, `one`, `peace`, `three`, `palm` (negative: should map to no action).
  `three` is index+middle+ring in HaGRID; `three2` (thumb+index+middle) has no gesture in the controller and is excluded. Images are not stored in this repo (`fetch_data.py` downloads them).
- Landmarks: MediaPipe Hands 0.10.14, `static_image_mode`, `min_detection_confidence=0.7` (the controller's setting). 259 of 299 readable images produced a hand (86.6%); undetected hands are skipped for the accuracy numbers.
- Classifiers (`gesture_bench/classifiers.py`):
  - `original`: the controller's rule: a finger is up if its tip is above its joint in image coordinates; count the fingers.
  - `improved`: a finger is extended if its tip is farther from the wrist than its middle joint (any rotation), and the gesture needs an exact set of fingers (index only, index+middle, index+middle+ring, none).
- The extension ratio (1.1) was picked on the even-numbered half of the detected images; the headline number is the other half (129 images).

## Results
| | original | improved |
|---|---|---|
| held-out half (129 images) | **78.3%** | **87.6%** |
| all 259 detected hands | 79.5% | 88.8% |

Per class (all detected): fist 100% -> 94%, one 76% -> 80%, peace 88% -> 98%, three 63% -> 75%, palm (should be ignored) 72% -> 98%.
The improved rules lose a little on the fist and gain most on the negative class: the old rule fires a game action on an open palm 28% of the time.
MediaPipe Hands runs at about 18 ms per 384 px image on 2 CPU cores (p95 19 ms), so about 50 FPS of headroom; the classifier itself is microseconds.
Honest limits: HaGRID is not webcam footage of one person playing a game, only static frames, and 129 test images is a small test set. `tests/` check the rotation invariance on synthetic hands.

```
pip install mediapipe==0.10.14 opencv-python remotezip && python3 fetch_data.py && python3 evaluate.py
python3 tests/test_classifiers.py
```

## Questions you will get
- *Why did the old rules fail?* They compare y coordinates, so a tilted or rotated hand reads fingers as folded; they also count fingers without checking which ones, so index+pinky counts as "two fingers".
- *Why a distance-from-wrist test?* It is invariant to in-plane rotation and uses landmarks MediaPipe already produces; no training data needed.
- *How do you avoid overfitting the threshold?* One parameter chosen on half the data and reported on the other half.
- *What would you do next?* Record real play sessions, label them, measure end-to-end latency from camera frame to key event, and add temporal smoothing (a gesture must hold for N frames) to cut false fires.
