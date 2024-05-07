import os
import easyocr
import cv2
from matplotlib import pyplot as plt


class EasyocrExtract:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def recognize_text(self, img):

        result = self.reader.readtext(img)

        # Filter to get only bounding boxes and text
        filtered_result = [(bbox, text) for bbox, text, *_ in result]
        return filtered_result

    def draw_ocr_boxes(self, img, save_name):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dpi = 80
        fig_width, fig_height = int(img.shape[0] / dpi), int(img.shape[1] / dpi)
        f, axarr = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        axarr[0].imshow(img)

        result = self.recognize_text(img)

        # Overlay bounding boxes for text
        for (bbox, text) in result:
            top_left, top_right, bottom_right, bottom_left = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

            cv2.rectangle(img, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=10)
        axarr[1].imshow(img)



    def extract_text(self, img):
        result = self.recognize_text(img)

        # Print extracted text only
        for bbox, text in result:
            print(text)


