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
                "label": ann['best_guess_labels'],
                "web_description": ann['web_description'],
                'page_title': ann['page_title_matching_images']
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
        
        
    def get_instruction(self, label, title):
        prompt = f"""
        You are a helpful assistant specializing in writing image captions for improved accuracy and informativeness.
        You will be provided with an image caption along with additional context in the following format:
        <context>
        LABEL: {label}  
        RELATED PAGE TITLE: {title}  
        </context>  

        Your task is to integrate all of the key details from the context into the image while maintaining clarity and relevance.
        Ensure the caption is more precise and informative.
        Output only the caption, without any extra text.
    """

        return prompt

    
    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["img_path"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        text_input = self.text_processor( self.get_instruction( ann["best_guess_labels"], ann['page_title_matching_images'] ) )
        caption = self.text_processor(ann['rewritten_caption'])

        return {
            "image": image,
            "text_input": text_input,
            "text_output": caption,
            "image_id": self.img_ids[ann["idx"]],
        }