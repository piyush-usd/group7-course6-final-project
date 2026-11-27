
from pathlib import Path

from load_dataset import load_json
from filter_classes import filter_classes
from validate_and_download_images import process_split
from create_csvs import build_split_csv
from create_classes import generate_classes_json

DATA_PATH = Path("../../data")
SPLIT_PATH = DATA_PATH / "splits"

SPLIT_PATH.mkdir(parents=True, exist_ok=True)


DATA_PATH = Path("../../data")
SPLIT_PATH = DATA_PATH / "splits"

SPLIT_PATH.mkdir(parents=True, exist_ok=True)

# 1. Generate classes.json first
generate_classes_json(
    input_file=DATA_PATH / "train_data.json",
    output_file=DATA_PATH/"classes.json"
)

# Load datasets
train_data = load_json(DATA_PATH / "train_data.json")
val_data = load_json(DATA_PATH / "val_data.json")
test_data = load_json(DATA_PATH / "test_data.json")

print("Train records>>>>>>>>>>>>>>>>>>>>:", len(train_data))
print("Validation records>>>>>>>>>>>>>>>>>>>:", len(val_data))
print("Test records:>>>>>>>>>>>>>>>>>>>", len(test_data))


# Filter selected classes
train_data = filter_classes(train_data)
val_data = filter_classes(val_data)
test_data = filter_classes(test_data)

print("Train records>>>>>>****After Filter*******>>>>>>>>>>>>>>:", len(train_data))
print("Validation records>>>>>>>******After Filter*****>>>>>>>>>>>>:", len(val_data))
print("Test records:>>>>>>>>>******After Filter*****>>>>>>>>>>", len(test_data))

# Validate images
train_data = process_split(train_data, "train",100)
val_data = process_split(val_data, "val",50)
test_data = process_split(test_data, "test",80)

print("Train records>>>>>>***After Validate and download Image***>>>>=========:", len(train_data))
print("Validation records>>>>>>>****After Validate and download Image***>>>>>>>>=======:", len(val_data))
print("Test records:>>>>>>>>>**After Validate and download  Image**>>>>>========", len(test_data))

# Create CSV files
build_split_csv(train_data, "train")
build_split_csv(val_data, "val")
build_split_csv(test_data, "test")

print("\nAll dataset splits created successfully!")
