import os, cv2, numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# align_vgg2_test112.py  – re-run me once to create 112×112 crops
#!/usr/bin/env python3
# align_vgg2_test112_with_subset.py
#   – creates full aligned test-set *and* a separate aligned subset
import os, cv2, tqdm, random
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ─── CONFIG ──────────────────────────────────────────────────────────────
SRC  = r"D:/research project/implentation/datasets/raw_test"
DST  = r"D:/research project/implentation/datasets/raw_test_160"
SUB  = r"d:/research project/implentation/datasets/small_raw_test_160"

os.makedirs(DST, exist_ok=True)
os.makedirs(SUB, exist_ok=True)

# ─── HOW TO CHOOSE THE SUBSET ────────────────────────────────────────────
# (uncomment ONE of the three blocks below)

# 1️⃣  TXT file with specific IDs ───────────────
# with open("subset_id_list.txt") as f:
#     keep_ids = set(id_.strip() for id_ in f if id_.strip())

# 2️⃣  First N IDs in alphabetical order ────────
#N = 20                          # <-- change if you like
#keep_ids = set(sorted(os.listdir(SRC))[:N])

# 3️⃣  Random percentage of identities/images ───
PCT = 0.10                   # keep 10 %
keep_ids = None              # we’ll decide per-image later

# ─── COUNT ALL IMAGES FIRST ──────────────────────────────────────────────
total_images = sum(
    len(files)
    for _, _, files in os.walk(SRC)
    if files
)
print(f"Found {total_images:,} images to process.")

# ─── INITIALISE INSIGHTFACE ─────────────────────────────────────────────
app = FaceAnalysis(name="buffalo_l",
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))    # ctx_id=-1 → CPU only

# ─── MAIN LOOP WITH PROGRESS BAR ────────────────────────────────────────
with tqdm.tqdm(total=total_images, desc="Processing images") as pbar:
    for person in os.listdir(SRC):
        src_person_dir = os.path.join(SRC, person)
        dst_person_dir = os.path.join(DST, person)
        sub_person_dir = os.path.join(SUB, person)

        os.makedirs(dst_person_dir, exist_ok=True)
        if (keep_ids and person in keep_ids) or keep_ids is None:
            os.makedirs(sub_person_dir, exist_ok=True)

        for img_name in os.listdir(src_person_dir):
            src_path = os.path.join(src_person_dir, img_name)
            dst_path = os.path.join(dst_person_dir, img_name)
            sub_path = os.path.join(sub_person_dir, img_name)

            img = cv2.imread(src_path)
            if img is None:
                pbar.update()
                continue

            faces = app.get(img)
            if not faces:
                pbar.update()
                continue

            face = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
            aligned = face_align.norm_crop(img, face.kps, 224)
            cv2.imwrite(dst_path, aligned)

            # Should this go into the subset?
            copy_to_subset = (
                (keep_ids  and person in keep_ids)           or   # cases 1 & 2
                (keep_ids is None and random.random() < PCT)       # case 3
            )
            if copy_to_subset:
                cv2.imwrite(sub_path, aligned)

            pbar.update()