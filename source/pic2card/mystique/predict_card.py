"""Module to  get the predicted adaptive card json"""

import json
import os
from typing import Dict, List, Tuple
import base64
import io
from io import BytesIO

import cv2
from PIL import Image
import numpy as np
import requests

from mystique.arrange_card import CardArrange
from mystique.extract_properties import ExtractProperties
from mystique.image_extraction import ImageExtraction
from mystique.card_template import DataBinding
from mystique import config


class PredictCard:
    """
    Collects the faster rcnn detected objects and handles the
    functionality of calling diffrent modules in card prediction
    and returning the predicted json objects.
    """

    def __init__(self, od_model=None):
        """
        Find the card components using Object detection model
        """
        self.od_model = od_model

    def collect_objects(self, output_dict=None, pil_image=None):
        """
        Returns the design elements from the faster rcnn model with its
        properties mapped

        @param output_dict: output dict from the object detection

        @return: Collected json of the design objects
                 and list of detected object's coordinates
        """
        extract_properties = ExtractProperties()
        boxes = output_dict["detection_boxes"]
        scores = output_dict["detection_scores"]
        classes = output_dict["detection_classes"]
        r, c = boxes.shape
        detected_coords = []
        json_object = {}.fromkeys(["objects"], [])
        width, height = pil_image.size
        # object - class mapping
        object_class_mapping = {
            "1": "textbox",
            "2": "radiobutton",
            "3": "checkbox",
            "4": "actionset",
            "5": "image"
        }
        for i in range(r):
            if scores[i] * 100 >= 90.0:
                object_json = dict().fromkeys(
                    ["object", "xmin", "ymin", "xmax", "ymax"], "")
                object_json["object"] = object_class_mapping.get(
                    str(classes[i]), None)
                if object_json["object"]:
                    ymin = boxes[i][0] * height
                    xmin = boxes[i][1] * width
                    ymax = boxes[i][2] * height
                    xmax = boxes[i][3] * width

                    object_json["xmin"] = float(xmin)
                    object_json["ymin"] = float(ymin)
                    object_json["xmax"] = float(xmax)
                    object_json["ymax"] = float(ymax)
                    object_json["coords"] = ",".join([str(xmin),
                                                      str(ymin), str(xmax),
                                                      str(ymax)])
                    object_json["score"] = scores[i]
                    object_json["horizontal_alignment"] = extract_properties.get_alignment(
                        image=pil_image, xmin=float(xmin), xmax=float(xmax))
                    object_json["data"] = extract_properties.get_text(
                        image=pil_image, coords=(xmin, ymin, xmax, ymax))

                    if object_json["object"] == "actionset":
                        object_json["style"] = extract_properties.get_actionset_type(
                            image=pil_image, coords=(xmin, ymin, xmax, ymax))

                    if object_json["object"] == "textbox":
                        detected_coords.append(
                            (xmin - 5, ymin, xmax + 5, ymax))
                        object_json["size"], object_json["weight"] = extract_properties.get_size_and_weight(
                            image=pil_image, coords=(xmin, ymin, xmax, ymax))
                        object_json["color"] = extract_properties.get_colors(
                            image=pil_image, coords=(xmin, ymin, xmax, ymax))

                    else:
                        detected_coords.append((xmin, ymin, xmax, ymax))

                    if object_json["object"] == "image":
                        cropped = pil_image.crop((xmin, ymin, xmax, ymax))
                        buff = BytesIO()
                        cropped.save(buff, format="PNG")
                        base64_string = base64.b64encode(
                            buff.getvalue()).decode()
                        object_json["data"] = f'data:image/png;base64,{base64_string}'
                        img_width, img_height = cropped.size
                        object_json["size"] = (
                            round(img_width), round(img_height))

                    json_object["objects"].append(object_json)
        return json_object, detected_coords

    def main(self, image=None, card_format=None):
        """
        Handles the different components calling and returns the
        predicted card json to the API

        @param labels_path: faster rcnn model's label path
        @param forzen_graph_path: faster rcnn model path
        @param image: input image path

        @return: predicted card json
        """
        image = image.convert("RGB")
        image_np = np.asarray(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # Extract the design objects from faster rcnn model
        output_dict, category_index = self.od_model.get_objects(image=image_np)
        return self.generate_card(output_dict, image, image_np, card_format)

    def tf_serving_main(self, bs64_img: str, tf_server: str, model_name: str,
                        card_format: str = None) -> Dict:
        tf_uri = os.path.join(tf_server,
                              f"v1/models/{model_name}:predict")
        payloads = {
            "signature_name": "serving_default",
            "instances": [
                {"b64": bs64_img}
            ]
        }

        # Hit the tf-serving and get the prediction
        response = requests.post(tf_uri, json=payloads)
        pred_res = json.loads(response.content)["predictions"][0]

        filtered_res = {}
        for key_col in ["detection_boxes", "detection_scores",
                        "detection_classes"]:
            filtered_res[key_col] = np.array(pred_res[key_col])

        # Prepare the card from object detection.
        pic2card = PredictCard(None)
        imgdata = base64.b64decode(bs64_img)
        image = Image.open(io.BytesIO(imgdata))
        image = image.convert("RGB")
        image_np = np.asarray(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        card = pic2card.generate_card(filtered_res,
                                      image, image_np, card_format)
        return card

    def generate_card(self, prediction: Dict, image: Image,
                      image_np: np.array, card_format: str):
        """
        From the object detection result and image, generate adaptive
        card object.

        @param prediction: Prediction result from rcnn model
        @param image: PIL Image object to crop the regions.
        @param image_np: Array representation of the image.
        @param card_format: format specification for template data binding

        """
        # TODO: Remove the reduendant usage of image and image_np

        # Collect the objects along with its design properites

        json_objects, detected_coords = self.collect_objects(
            output_dict=prediction, pil_image=image)
        # Remove overlapping rcnn objects
        card_arrange = CardArrange()
        card_arrange.remove_overlapping_objects(json_object=json_objects)

        # Get image coordinates from custom image pipeline
        if config.USE_CUSTOM_IMAGE_PIPELINE:
            self.get_image_objects(json_objects, detected_coords, card_arrange,
                                   image_np, image)
        # Arrange the design elements
        return_dict = {}.fromkeys(["card_json"], "")
        card_json = {
            "type": "AdaptiveCard",
            "version": "1.0",
            "body": [],
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json"
        }

        body, ymins = card_arrange.build_card_json(
            objects=json_objects.get("objects", []))
        # Sort the elements vertically
        body = [x for _, x in sorted(zip(ymins, body),
                                     key=lambda x: x[0])]
        # if format==template - generate template data json
        if card_format == "template":
            databinding = DataBinding()
            data_payload, body = databinding.build_data_binding_payload(body)
            return_dict["card_v2_json"] = {}.fromkeys(["data", "template"], {})
            return_dict["card_v2_json"]["data"] = data_payload
        # Prepare the response with error code
        error = None
        if not body or not detected_coords:
            error = {
                "msg": "Failed to generate card components",
                "code": 1000
            }
        else:
            card_json["body"] = body

        if card_format != "template":
            return_dict["card_json"] = card_json
        else:
            return_dict["card_v2_json"]["template"] = card_json
            return_dict["card_json"] = None
        return_dict["error"] = error

        return return_dict

    def get_image_objects(self, json_objects: List[Dict],
                          detected_coords: List[Tuple],
                          card_arrange: CardArrange, image_np: np.array,
                          image: Image):
        """
        Collects the image objects using the custom image pipeline

        @param json_objects: list of design objects from the rcnn model
        @param detected_coords: list of coordinates of the design objects
        @param card_arrange: CardArrange object
        @param image_np: opencv input image
        @param image: PIL input image
        """
        # delete image from the rcnn model detected objects and coordinates
        positions_to_remove = [ctr for ctr, design_object in enumerate(
            json_objects.get("objects", [])) if design_object.get("object") == "image"]
        json_objects["objects"] = [design_object for ctr, design_object in enumerate(
            json_objects.get("objects")) if ctr not in positions_to_remove]
        detected_coords = [coords for ctr, coords in enumerate(
            detected_coords) if ctr not in positions_to_remove]

        image_extraction = ImageExtraction()
        image_points = image_extraction.detect_image(image=image_np,
                                                     detected_coords=detected_coords,
                                                     pil_image=image)
        image_urls, image_sizes = image_extraction.image_crop_get_url(
            coords=image_points,
            image=image)
        # Append the image objects to the deisgn objects list
        card_arrange.append_image_objects(image_urls=image_urls,
                                          image_sizes=image_sizes,
                                          image_coords=image_points,
                                          pil_image=image,
                                          json_object=json_objects)
