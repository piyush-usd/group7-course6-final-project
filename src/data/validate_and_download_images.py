import requests

from pathlib import Path
from io import BytesIO
from collections import Counter
from PIL import Image, UnidentifiedImageError


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
DATA_DIR = Path("../../data")
IMAGE_DIR = DATA_DIR / "images"

# --------------------------------------------------
# 2. Validate and download one image
# --------------------------------------------------

def validate_and_download(item, output_dir, session):

    image_url = item.get("image_url", "").strip()
    category = item.get("class_label", "").strip()

    if not image_url:
        return None, "Missing image URL"

    try:
        response = session.get(
            image_url,
            timeout=20
        )

        response.raise_for_status()

        image_bytes = response.content

        # Validate image content in memory
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()

        # Reopen and fully decode the image
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()

        # Save only after successful validation
        original_path = Path(item.get("image_path", ""))
        filename = original_path.name

        if not filename:
            return None, "Missing image filename"

        local_path = output_dir / filename

        with open(local_path, "wb") as image_file:
            image_file.write(image_bytes)

        return {
            "image_path": str(local_path.relative_to(DATA_DIR)),
            "image_url": image_url,
            "class_label": category,
            "label_idx": item["label_idx"]
        }, None

    except (
        requests.RequestException,
        OSError,
        ValueError,
        UnidentifiedImageError
    ) as error:

        return None, str(error)


# --------------------------------------------------
# 3. Process dataset and return downloaded records
# --------------------------------------------------

def process_split(filtered_data, folder_name, limit=None):

    data_to_process = filtered_data[:limit] if limit is not None else filtered_data

    output_dir = IMAGE_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_records = []
    dropped_counts = Counter()

    print(f"\nProcessing dataset -> {output_dir}")
    print(f"Records to process: {len(data_to_process)} (out of {len(filtered_data)})")

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "Mozilla/5.0"
        })

        for index, item in enumerate(data_to_process, start=1):

            record, error = validate_and_download(
                item,
                output_dir,
                session
            )

            if record:
                downloaded_records.append(record)
            else:
                category = item.get("class_label", "Unknown")
                dropped_counts[category] += 1
                print(f"Skipped {index}: {category} | {error}")

            if index % 100 == 0:
                print(f"Processed {index}/{len(data_to_process)}")

    # Summary
    print(f"\n========== {folder_name.upper()} DOWNLOAD REPORT ==========")
    print("Total processed records:", len(data_to_process))
    print("Successfully downloaded:", len(downloaded_records))
    print("Failed records:", sum(dropped_counts.values()))

    print("\nFailed images per class:")
    for category, count in sorted(dropped_counts.items()):
        print(f"{category}: {count}")
    print("==============================================")

    return downloaded_records
