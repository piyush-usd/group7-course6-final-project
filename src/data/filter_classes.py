
import json

DATA_PATH = "../processed/"

include_classes = {
    "Saree",
    "Lehenga",
    "women_kurta",
    "leggings_and_salwar",
    "palazzo",
    "dupattas",
    "blouse",
    "gowns",
    "dhoti_pants",
    "petticoats"
}

# Convert selected classes to lowercase
include_classes = {
    category.strip().lower()
    for category in include_classes
}

print("DATA_PATH mapping:", DATA_PATH)

# Load class mapping
with open(DATA_PATH + "classes.json", "r") as file:
    class_data = json.load(file)


label_idx = {
    item["category"].strip().lower(): item["id"]
    for item in class_data["classes"]
}


def filter_classes(data):

    filtered_data = []

    for item in data:

        category = item.get("class_label", "").strip()

        if category.lower() in include_classes and category.lower() in label_idx:
            item["label_idx"] = label_idx[category]
            filtered_data.append(item)

    return filtered_data


print("Class mapping:", label_idx)