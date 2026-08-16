"""
Table structure recognition using Microsoft's Table Transformer (TATR).
Detects rows, columns, and individual cells from an already-cropped table image.

Install first:
    pip install transformers torch torchvision pillow --break-system-packages
"""

from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from PIL import Image, ImageDraw
import torch


def load_model():
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    return processor, model


def detect_structure(image_path, result_path="tatr_result.png", score_threshold=0.6):
    image = Image.open(image_path).convert("RGB")

    processor, model = load_model()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert model output to boxes in original image coordinates
    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(
        outputs, threshold=score_threshold, target_sizes=target_sizes
    )[0]

    # id2label tells you what each detected box actually is:
    # 'table', 'table row', 'table column', 'table column header',
    # 'table projected row header', 'table spanning cell'
    id2label = model.config.id2label

    rows = []
    columns = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        label = id2label[label_id.item()]
        box = [round(v) for v in box.tolist()]  # [xmin, ymin, xmax, ymax]

        if label == "table row":
            rows.append(box)
        elif label == "table column":
            columns.append(box)

    # Derive individual cells as the intersection of each row x column
    cells = []
    for r in rows:
        r_ymin, r_ymax = r[1], r[3]
        for c in columns:
            c_xmin, c_xmax = c[0], c[2]
            cells.append([c_xmin, r_ymin, c_xmax, r_ymax])

    # Visualize
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for box in rows:
        draw.rectangle(box, outline="red", width=2)
    for box in columns:
        draw.rectangle(box, outline="blue", width=2)
    for box in cells:
        draw.rectangle(box, outline="green", width=1)

    output.save(result_path)

    print(f"Detected {len(rows)} rows, {len(columns)} columns, {len(cells)} cells")
    print(f"Saved visualization to: {result_path}")

    return rows, columns, cells


if __name__ == "__main__":
    detect_structure("../cropped_bill.png")