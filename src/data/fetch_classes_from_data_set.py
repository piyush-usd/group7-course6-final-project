
import json

file_path = "../../data/train_data.json"

classes = set()

# Output JSON file
output_file = "../processed/classes.json"

class Fashion:
    def __init__(self, id, category):
        self.id = id
        self.category = category

    def display(self):
        print("Id:", self.id)
        print("Category:", self.category)

    def to_dict(self):
        return {
            "id": self.id,
            "category": self.category
        }


with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        item = json.loads(line)

        # Extract class label
        if "class_label" in item:
            classes.add(item["class_label"])

print("Available Classes:")

# Create Fashion objects
fashion_classes = []


for index, class_name in enumerate(sorted(classes), start=1):
    fashion = Fashion(index, class_name.strip())
    fashion_classes.append(fashion)
    fashion.display()

print(f"\nTotal classes: {len(classes)}")

# Create JSON data
json_data = {
    "classes": [fashion.to_dict() for fashion in fashion_classes]
}

# Write JSON file
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(json_data, file, indent=4, ensure_ascii=False)

print(f"\nJSON file created successfully: {output_file}")
