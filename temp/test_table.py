# test_table.py
import cv2
from core.domain.image_payload import ImagePayload
from table_extraction.extractors import grid_line_extractor  # triggers registration
from table_extraction.registry import extractor_registry

extractor = extractor_registry.create("grid_line", save_debug_images=True)

img = cv2.imread("temp/ty1.jpg")
result = extractor.extract(ImagePayload(image=img))

print(f"Tables found: {len(result.tables)}")
for t in result.tables:
    print(f"  Table {t.table_id[:8]}: {len(t.cells)} cells, bbox={t.bbox}")