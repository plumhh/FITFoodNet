import os
import shutil

import pandas as pd
from PIL import Image, ImageOps


PRED_CSV = "/home/amax/4t/lzh/DINOv3/outputs/error_analysis/foodx251_val_predictions.csv"
PAIR_CSV = "/home/amax/4t/lzh/DINOv3/outputs/error_analysis/foodx251_top_confused_pairs.csv"

OUT_DIR = "/home/amax/4t/lzh/DINOv3/outputs/error_analysis/failure_case_candidates"
RAW_DIR = os.path.join(OUT_DIR, "raw")
SQUARE_DIR = os.path.join(OUT_DIR, "square_320")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SQUARE_DIR, exist_ok=True)


# Maximum examples exported for each directed confusion pair.
TOP_K_PER_PAIR = 8

# Number of pairs loaded from the ranked confusion-pair CSV.
NUM_TOP_PAIRS = 25

# Additional globally highest-confidence errors to export.
NUM_GLOBAL_HIGH_CONF = 50


# Priority confusion pairs are exported before automatically ranked pairs.
PRIORITY_PAIRS = [
    ("oyster", "huitre"),
    ("huitre", "oyster"),
    ("steak_tartare", "beef_tartare"),
    ("beef_tartare", "steak_tartare"),
    ("chicken_wing", "buffalo_wing"),
    ("buffalo_wing", "chicken_wing"),
    ("sushi", "sashimi"),
    ("sashimi", "sushi"),
    ("barbecued_spareribs", "baby_back_rib"),
    ("baby_back_rib", "barbecued_spareribs"),
]


def safe_name(text):
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def load_pairs():
    pairs = []

    # Start with manually selected priority pairs.
    for pair in PRIORITY_PAIRS:
        pairs.append(pair)

    # Append ranked pairs from the confusion analysis when available.
    if os.path.exists(PAIR_CSV):
        pair_df = pd.read_csv(PAIR_CSV)
        for _, row in pair_df.head(NUM_TOP_PAIRS).iterrows():
            pair = (row["true_label"], row["pred_label"])
            if pair not in pairs:
                pairs.append(pair)
    else:
        print(f"Warning: pair CSV not found: {PAIR_CSV}")

    return pairs


def export_one_image(src, dst_raw, dst_square):
    shutil.copy2(src, dst_raw)

    img = Image.open(src).convert("RGB")
    img_square = ImageOps.fit(
        img,
        (320, 320),
        method=Image.Resampling.LANCZOS,
    )
    img_square.save(dst_square, quality=95)


def export_pair_cases(df, pairs):
    wrong = df[df["true_label"] != df["pred_label"]].copy()
    wrong["confidence"] = wrong["confidence"].astype(float)

    records = []
    used_paths = set()
    export_index = 1

    for pair_idx, (gt, pred) in enumerate(pairs, start=1):
        candidates = wrong[
            (wrong["true_label"] == gt) & (wrong["pred_label"] == pred)
        ].copy()

        if len(candidates) == 0:
            print(f"Warning: no cases found for {gt} -> {pred}")
            continue

        candidates = candidates.sort_values("confidence", ascending=False)

        pair_folder_name = f"{pair_idx:02d}_{safe_name(gt)}_to_{safe_name(pred)}"
        raw_pair_dir = os.path.join(RAW_DIR, pair_folder_name)
        square_pair_dir = os.path.join(SQUARE_DIR, pair_folder_name)

        os.makedirs(raw_pair_dir, exist_ok=True)
        os.makedirs(square_pair_dir, exist_ok=True)

        count = 0

        for _, row in candidates.iterrows():
            if count >= TOP_K_PER_PAIR:
                break

            src = row["image_path"]
            if not os.path.exists(src):
                continue

            if src in used_paths:
                continue

            conf = float(row["confidence"])
            ext = os.path.splitext(src)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                ext = ".jpg"

            base_name = (
                f"{export_index:04d}_GT-{safe_name(gt)}_"
                f"Pred-{safe_name(pred)}_Conf-{conf:.2f}"
            )

            raw_dst = os.path.join(raw_pair_dir, base_name + ext)
            square_dst = os.path.join(square_pair_dir, base_name + ".jpg")

            export_one_image(src, raw_dst, square_dst)

            records.append(
                {
                    "index": export_index,
                    "source": "pair",
                    "pair_rank": pair_idx,
                    "true_label": gt,
                    "pred_label": pred,
                    "confidence": conf,
                    "source_path": src,
                    "raw_image": raw_dst,
                    "square_320_image": square_dst,
                }
            )

            used_paths.add(src)
            export_index += 1
            count += 1

        print(f"Exported {count} cases for {gt} -> {pred}")

    return records, used_paths, export_index


def export_global_high_conf_cases(df, used_paths, start_index):
    wrong = df[df["true_label"] != df["pred_label"]].copy()
    wrong["confidence"] = wrong["confidence"].astype(float)
    wrong = wrong.sort_values("confidence", ascending=False)

    raw_global_dir = os.path.join(RAW_DIR, "global_high_conf_errors")
    square_global_dir = os.path.join(SQUARE_DIR, "global_high_conf_errors")

    os.makedirs(raw_global_dir, exist_ok=True)
    os.makedirs(square_global_dir, exist_ok=True)

    records = []
    export_index = start_index
    count = 0

    for _, row in wrong.iterrows():
        if count >= NUM_GLOBAL_HIGH_CONF:
            break

        src = row["image_path"]
        if not os.path.exists(src):
            continue

        if src in used_paths:
            continue

        gt = row["true_label"]
        pred = row["pred_label"]
        conf = float(row["confidence"])

        ext = os.path.splitext(src)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            ext = ".jpg"

        base_name = (
            f"{export_index:04d}_GT-{safe_name(gt)}_"
            f"Pred-{safe_name(pred)}_Conf-{conf:.2f}"
        )

        raw_dst = os.path.join(raw_global_dir, base_name + ext)
        square_dst = os.path.join(square_global_dir, base_name + ".jpg")

        export_one_image(src, raw_dst, square_dst)

        records.append(
            {
                "index": export_index,
                "source": "global_high_conf",
                "pair_rank": "",
                "true_label": gt,
                "pred_label": pred,
                "confidence": conf,
                "source_path": src,
                "raw_image": raw_dst,
                "square_320_image": square_dst,
            }
        )

        used_paths.add(src)
        export_index += 1
        count += 1

    print(f"Exported {count} global high-confidence error cases.")

    return records


def main():
    if not os.path.exists(PRED_CSV):
        raise FileNotFoundError(f"Prediction CSV not found: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV)

    required_cols = ["image_path", "true_label", "pred_label", "confidence"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    pairs = load_pairs()

    print("Selected confusing pairs:")
    for i, (gt, pred) in enumerate(pairs, start=1):
        print(f"{i:02d}. {gt} -> {pred}")

    pair_records, used_paths, next_index = export_pair_cases(df, pairs)
    global_records = export_global_high_conf_cases(df, used_paths, next_index)

    all_records = pair_records + global_records

    label_csv = os.path.join(OUT_DIR, "failure_case_candidates_labels.csv")
    pd.DataFrame(all_records).to_csv(label_csv, index=False, encoding="utf-8")

    print("\nDone.")
    print(f"Total exported images: {len(all_records)}")
    print(f"Raw images saved to: {RAW_DIR}")
    print(f"Square images saved to: {SQUARE_DIR}")
    print(f"Label CSV saved to: {label_csv}")


if __name__ == "__main__":
    main()
