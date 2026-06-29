from pathlib import Path
from src.data.manifest import create_dataset_manifest

DATASET_PATH = Path(
    "data/raw/Poultry Birds Poo Imagery Dataset for Health Status Prediction A Case of South-West Nigeria/Dataset"
)


OUTPUT_DIR = Path("artifacts/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

manifest = create_dataset_manifest(DATASET_PATH)

manifest.to_csv(OUTPUT_DIR / "dataset_manifest", index=False)


print("\n" + "=" * 60)
print("DATASET MANIFEST")
print("=" * 60)

print(f"Total image: {len(manifest)}")
print(manifest.tail())
