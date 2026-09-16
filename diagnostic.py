import os

BASE_DIR = r"D:\Thesis"

DATASETS = {
    "TRAINING": os.path.join(
        BASE_DIR,
        "Training and Validation Set",
    ),
    "TEST": os.path.join(
        BASE_DIR,
        "Held out Test set"
    )
}

for split_name, split_dir in DATASETS.items():
    print(f"\n{'=' * 60}")
    print(f"{split_name}: {split_dir}")
    print(f"{'=' * 60}")

    for dataset_type in ["Real_Dataset", "AI_Dataset"]:
        base = os.path.join(split_dir, dataset_type)

        print(f"\n--- {dataset_type} ---")

        if not os.path.exists(base):
            print(f"NOT FOUND: {base}")
            continue

        total = 0

        speakers = sorted(
            folder for folder in os.listdir(base)
            if os.path.isdir(os.path.join(base, folder))
        )

        for speaker in speakers:
            speaker_path = os.path.join(base, speaker)

            count = sum(
                1
                for f in os.listdir(speaker_path)
                if f.lower().endswith(".wav")
            )

            total += count
            print(f"{speaker}: {count} audio files")

        print(f"TOTAL {dataset_type}: {total}")

    print(f"\n{'-' * 60}")