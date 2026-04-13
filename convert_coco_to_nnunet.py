import os
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO


# =========================
# 你只要改這裡
# =========================
PROJECT_DIR = Path(r"D:\ANFC_project")
TRAIN_DIR = PROJECT_DIR / "train"
TRAIN_JSON = TRAIN_DIR / "_annotations.coco.json"
DATASET_NAME = "Dataset501_ANFC"
SKIP_CATEGORY_NAMES = {"55"}   # 你的 supercategory
# =========================


def build_label_mapping(coco: COCO, skip_names=None):
    if skip_names is None:
        skip_names = set()

    categories = sorted(coco.loadCats(coco.getCatIds()), key=lambda x: x["id"])

    labels_dict = {"background": 0}
    coco_catid_to_label = {}
    label_id = 1

    for cat in categories:
        name = str(cat["name"]).strip()
        if name in skip_names:
            continue
        labels_dict[name] = label_id
        coco_catid_to_label[cat["id"]] = label_id
        label_id += 1

    return labels_dict, coco_catid_to_label


def main():
    nnunet_raw_dir = Path(os.environ.get("nnUNet_raw", str(PROJECT_DIR / "nnUNet_raw")))
    dataset_dir = nnunet_raw_dir / DATASET_NAME
    imagesTr_dir = dataset_dir / "imagesTr"
    labelsTr_dir = dataset_dir / "labelsTr"

    imagesTr_dir.mkdir(parents=True, exist_ok=True)
    labelsTr_dir.mkdir(parents=True, exist_ok=True)

    if not TRAIN_JSON.exists():
        raise FileNotFoundError(f"CAN GET TRAIN: {TRAIN_JSON}")

    coco = COCO(str(TRAIN_JSON))

    labels_dict, coco_catid_to_label = build_label_mapping(
        coco,
        skip_names=SKIP_CATEGORY_NAMES
    )

    img_ids = sorted(coco.getImgIds())
    converted_count = 0
    skipped_missing = 0
    skipped_bad = 0

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info["file_name"]
        img_path = TRAIN_DIR / img_name

        if not img_path.exists():
            print(f"[跳過] 找不到圖片: {img_path}")
            skipped_missing += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[跳過] 讀圖失敗: {img_path}")
            skipped_bad += 1
            continue

        # OpenCV 讀進來是 BGR，轉成 RGB 再存
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        case_id = f"ANFC_{converted_count:03d}"
        out_img_path = imagesTr_dir / f"{case_id}_0000.png"
        out_mask_path = labelsTr_dir / f"{case_id}.png"

        ok = cv2.imwrite(str(out_img_path), img_rgb)
        if not ok:
            print(f"[跳過] 寫入影像失敗: {out_img_path}")
            skipped_bad += 1
            continue

        h = int(img_info["height"])
        w = int(img_info["width"])
        mask = np.zeros((h, w), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in coco_catid_to_label:
                continue
            try:
                pixel_mask = coco.annToMask(ann)
                mask[pixel_mask > 0] = coco_catid_to_label[cat_id]
            except Exception as e:
                print(f"[警告] annToMask 失敗, ann_id={ann.get('id')}, error={e}")

        ok = cv2.imwrite(str(out_mask_path), mask)
        if not ok:
            print(f"[跳過] 寫入 mask 失敗: {out_mask_path}")
            skipped_bad += 1
            if out_img_path.exists():
                out_img_path.unlink(missing_ok=True)
            continue

        converted_count += 1

    dataset_json = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "labels": labels_dict,
        "numTraining": converted_count,
        "file_ending": ".png"
    }

    dataset_json_path = dataset_dir / "dataset.json"
    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)

    print("\n=== 完成 ===")
    print(f"輸出資料夾: {dataset_dir}")
    print(f"成功轉換: {converted_count}")
    print(f"缺圖跳過: {skipped_missing}")
    print(f"壞圖/寫檔失敗: {skipped_bad}")
    print(f"dataset.json: {dataset_json_path}")
    print("labels =", labels_dict)


if __name__ == "__main__":
    main()
