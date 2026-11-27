import json
from pathlib import Path


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


def generate_classes_json(
    input_file="../../data/train_data.json",
    output_file="../processed/classes.json"
):
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = set()

    with open(input_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            if "class_label" in item:
                classes.add(item["class_label"])

    print("Available Classes:")

    fashion_classes = []

    for index, class_name in enumerate(sorted(classes), start=1):
        fashion = Fashion(index, class_name.strip())
        fashion_classes.append(fashion)
        fashion.display()

    print(f"\nTotal classes: {len(classes)}")

    json_data = {
        "total_classes": len(fashion_classes),
        "classes": [fashion.to_dict() for fashion in fashion_classes]
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=4, ensure_ascii=False)

    print(f"\nJSON file created successfully: {output_path}\n")
    return json_data
