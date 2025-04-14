import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import json


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "image": sample["image_path"],
                "caption": ann["caption"],
                'rewritten_caption': ann['rewritten_caption'],
                "date": ann['date'],
                "location": ann['location'],
                'people': ann['people'],
                "event": ann['event'],
                "motivation": ann['motivation'],
                'object': ann['object']
            }
        )

class OOCDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:         
            img_id = ann["instance_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
        
    def get_instruction(self, date, location, people, event, motivation, object):
        prompt = f"""
            You are given a news photograph and optional contextual information including date, location, people, event, objects, and motivation.

            Your task is to analyze the image and write a caption of the event shown in the image.

            - Do not speculate or add information not present in the image or fields.
            - Avoid adjectives or descriptive storytelling.
            - Some fields may be missing.

            - Date: {date}
            - Location: {location}
            - People: {people}
            - Event: {event}
            - Objects: {object}
            - Motivation: {motivation}

            Summary:
            """
        return prompt

    
    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image_path"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        text_input = self.text_processor( self.get_instruction( ann['date'], ann['location'], ann['people'], ann['event'], ann['motivation'] , ann['object']  ) )
        caption = self.text_processor(ann['rewritten_caption'])

        return {
            "image": image,
            "text_input": text_input,
            "text_output": caption,
            "image_id": self.img_ids[ann["instance_id"]],
        }