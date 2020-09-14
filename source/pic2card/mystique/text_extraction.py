import numpy as np
import cv2

from mystique import default_host_configs
from pytesseract import pytesseract, Output


class TextExtraction:
    """
    Class handles extraction of text and properties like size and weight
    from the identified design elements.
    from all the design elements - extracts text
    from textual elements - extracts size, weight
    """

    def get_text(self, image=None, coords=None):
        """
        Extract the text from the object coordinates
        in the input deisgn image using pytesseract.
        @param image: input PIL image
        @param coords: tuple of coordinates from which
                       text should be extracted
        @return: ocr text , pytesseract image data
        """
        coords = (coords[0] - 5, coords[1], coords[2] + 5, coords[3])
        cropped_image = image.crop(coords)
        cropped_image = cropped_image.convert("LA")

        img_data = pytesseract.image_to_data(
            cropped_image, lang="eng", config="--psm 6",
            output_type=Output.DICT)
        text_list = ' '.join(img_data['text']).split()
        extracted_text = ' '.join(text_list)

        return extracted_text, img_data

    def get_size(self, image=None, coords=None, img_data=None):
        """
        Extract the size by taking an average of
        ratio of height of each character to height
        input image using pytesseract

        @param image : input PIL image
        @param coords: list of coordinated from which
                       text and height should be extracted
        @param img_data : input image data from pytesseract
        @return: size
        """
        image_width, image_height = image.size
        box_height = []
        n_boxes = len(img_data['level'])
        for i in range(n_boxes):
            if len(img_data['text'][i]) > 1:  # to ignore img with wrong bbox
                (_, _, _, h) = (img_data['left'][i], img_data['top'][i],
                                img_data['width'][i], img_data['height'][i])
                # h = text_size_processing(img_data['text'][i], h)

                box_height.append(h)
        font_size = default_host_configs.FONT_SIZE
        # Handling of unrecognized characters
        if len(box_height) == 0:
            heights_ratio = font_size['default']
        else:
            heights = int(np.mean(box_height))
            heights_ratio = round((heights/image_height), 4)

        if font_size['small'] < heights_ratio < font_size['default']:
            size = "Small"
        elif font_size['default'] < heights_ratio < font_size['medium']:
            size = "Default"
        elif font_size['medium'] < heights_ratio < font_size['large']:
            size = "Medium"
        elif font_size['large'] < heights_ratio < font_size['extralarge']:
            size = "Large"
        elif font_size['extralarge'] < heights_ratio:
            size = "ExtraLarge"
        else:
            size = "Default"

        return size

    def get_thickness(self, image=None, coords=None):
        """
        Extract the weight of the each words by
        finding the thickness of the font using
        skeletonizing the image

        @param image : input PIL image
        @param coords: list of coordinated from which
                       text and height should be extracted
        @return: weight
        """
        cropped_image = image.crop(coords)
        image_width, image_height = image.size
        c_img = np.asarray(cropped_image)
        """
        if(image_height/image_width) < 1:
            y_scale = round((800/image_width), 2)
            x_scale = round((500/image_height), 2)
            c_img = cv2.resize(c_img, (0, 0), fx=x_scale, fy=y_scale)
        """
        gray = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        area_of_img = np.count_nonzero(img)
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, open)
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        area_of_skel = np.sum(skel)/255
        thickness = round(area_of_img/area_of_skel, 2)

        font_weight = default_host_configs.FONT_WEIGHT_THICK

        if font_weight['lighter'] >= thickness:
            weight = "Lighter"
        elif font_weight['bolder'] <= thickness:
            weight = "Bolder"
        else:
            weight = "Default"

        return weight


class CollectTextProperties(TextExtraction):
    """
    Helps to collect the textual properties for respective design object.
    """

    def __init__(self, image=None):
        self.image = image

    def text_prop(self, image=None, coords=None):
        data, img_data = self.get_text(image=self.image, coords=coords)
        size = self.get_size(image=self.image,
                             coords=coords, img_data=img_data)
        weight = self.get_thickness(image=self.image, coords=coords)

        return data, size, weight
