from pathlib import Path
import json

from src.data.audit import get_dataset_summary

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


if __name__ == "__main__":
    main()
