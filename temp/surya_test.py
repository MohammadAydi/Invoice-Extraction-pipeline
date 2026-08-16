# this is a temporary test file.
import cv2
from core.domain.image_payload import ImagePayload
from ocr.engines import surya_engine  # triggers self-registration
from ocr.registry import engine_registry

engine = engine_registry.create("surya", langs=["ar"])

raw = cv2.imread("image.jpg")
payload = ImagePayload(image=raw)

result = engine.recognize(payload)

print(f"Engine: {result.engine_name}")
print(f"Fragments found: {len(result.fragments)}")
for frag in result.fragments[:10]:
    print(f"  [{frag.confidence:.2f}] '{frag.text}'  @ {frag.bbox}")