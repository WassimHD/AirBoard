

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch


img="/Users/wassim/Downloads/ocr_test.jpg"
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten", 
    use_fast=True
)

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

image = Image.open(img).convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
