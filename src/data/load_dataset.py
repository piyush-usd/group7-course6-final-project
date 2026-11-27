
import json

DATA_PATH = "../../data/"


def load_json(file_path):

    data = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:

            line = line.strip()

            if not line:
                continue

            data.append(json.loads(line))

    return data


train_data = load_json(DATA_PATH + "train_data.json")
val_data = load_json(DATA_PATH + "val_data.json")
test_data = load_json(DATA_PATH + "test_data.json")

print("Train records:", len(train_data))
print("Validation records:", len(val_data))
print("Test records:", len(test_data))
