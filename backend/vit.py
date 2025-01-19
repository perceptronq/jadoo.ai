import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests

class VisionTransformer():
    def __init__(self):
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    def imagetag(self, url):
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        boxes = outputs.pred_boxes
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = model.config.id2label[label.item()]
            print(f"Detected {label_name} with confidence {score.item():.2f} at {box.tolist()}")
        tags = {model.config.id2label[label.item()] for label in results["labels"]}
        print("Tags:", ", ".join(tags))
        return tags
