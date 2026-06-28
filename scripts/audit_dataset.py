from pathlib import Path
import json

from src.data.audit import (
    get_dataset_summary,
    find_corrupt_images,
    analyze_image_dimensions,
    analyze_image_statistics,
)

DATASET_PATH = Path(
    "data/raw/Poultry Birds Poo Imagery Dataset for Health Status Prediction A Case of South-West Nigeria/Dataset"
)


def main():

    summary = get_dataset_summary(DATASET_PATH)

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Dataset Path : {summary['dataset_path']}")
    print(f"Number of Classes : {summary['num_classes']}")
    print(f"Total Images : {summary['total_images']}")

    print("\nClasses")

    for class_name, count in summary["class_distribution"].items():
        print(f"  {class_name:<15} {count}")

    # save the summary report in json file
    path = Path("artifacts/reports/dataset_summary.json")

    with open(path, "w") as f:
        json.dump(summary, f)


# Save the corrupt image in the report path
REPORT_DIR = Path("artifacts/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

corrupt_df = find_corrupt_images(DATASET_PATH)

corrupt_df.to_csv(
    REPORT_DIR / "corrupt_images.csv",
    index=False,
)

print("\n" + "=" * 60)
print("CORRUPT IMAGE REPORT")
print("=" * 60)
print(f"Corrupt Images Found : {len(corrupt_df)}")


# Save the image dimension to the reports path
dimension_df = analyze_image_dimensions(DATASET_PATH)

dimension_df.to_csv(REPORT_DIR / "image_dimension.csv", index=False)

print("\n" + "=" * 60)
print("IMAGE DIMENSION REPORT")
print("=" * 60)
print(f"Image Analyzed: {len(dimension_df)}")
print(f"Minimum Width: {dimension_df['width'].min()}")
print(f"Maximum width: {dimension_df['width'].max()}")
print(f"Minimum height: {dimension_df['height'].min()}")
print(f"Maximum height: {dimension_df['height'].max()}")

print("\n", "Top 10 most common image size")

size_count = dimension_df[["width", "height"]].value_counts().head(10)

print(size_count)


# Save the image statistics and the rgb_pixels to the reports path
image_stats_df, rgb_stats_df = analyze_image_statistics(DATASET_PATH)

image_stats_df.to_csv(REPORT_DIR / "image_statistics.csv", index=False)
rgb_stats_df.to_csv(REPORT_DIR / "rgb_pixels.csv", index=False)

print("\n" + "=" * 60)
print("IMAGE STATISTICS REPORT")
print("=" * 60)

print(f"Format: {image_stats_df['format'].value_counts()}")
print(f"Mode: {image_stats_df['mode'].value_counts()}")
print(f"Average RGB Value {rgb_stats_df.mean().round(2)}")

if __name__ == "__main__":
    main()
