import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

image_url = "https://i.pinimg.com/originals/b9/af/27/b9af273ce170a622c1518a897e7d72e1.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
boxes = outputs.pred_boxes
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9
)[0]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    label_name = model.config.id2label[label.item()]
    print(f"Detected {label_name} with confidence {score.item():.2f} at {box.tolist()}")
tags = {model.config.id2label[label.item()] for label in results["labels"]}
print("Tags:", ", ".join(tags))

