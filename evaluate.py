"""python3 fetch_data.py && python3 evaluate.py   (needs mediapipe==0.10.x, opencv, remotezip)"""
import glob, json, statistics, time, pickle, os
import cv2, mediapipe as mp
from gesture_bench.classifiers import original, improved
EXPECT = {"fist": "Slide", "one": "Left", "peace": "Right", "three": "Jump", "palm": "Unknown"}   # 'three2' (thumb+index+middle) has no gesture in the controller and is excluded
hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
rows, lat, total = [], [], 0
for c in EXPECT:
    for p in sorted(glob.glob(f"data/{c}/*.jpg")):
        im = cv2.imread(p)
        if im is None: continue
        total += 1; t = time.perf_counter(); r = hands.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); lat.append((time.perf_counter() - t) * 1000)
        if r.multi_hand_landmarks: rows.append((c, [(l.x, l.y, l.z) for l in r.multi_hand_landmarks[0].landmark]))
acc = lambda rs, fn: sum(fn(l) == EXPECT[c] for c, l in rs) / len(rs)
test = rows[1::2]   # the ratio 1.1 was chosen on rows[0::2]; report the other half
res = dict(images=total, detected=len(rows), original_all=acc(rows, original), improved_all=acc(rows, improved), original_heldout=acc(test, original), improved_heldout=acc(test, improved),
           heldout_n=len(test), mediapipe_ms_median=statistics.median(lat), mediapipe_ms_p95=sorted(lat)[int(.95 * len(lat))],
           per_class={c: dict(n=sum(1 for k, _ in rows if k == c), original=acc([r for r in rows if r[0] == c], original), improved=acc([r for r in rows if r[0] == c], improved)) for c in EXPECT})
os.makedirs("results", exist_ok=True); json.dump(res, open("results/summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
