import cv2
from pytesseract import pytesseract, Output


class PytesseractExtract:
    def __init__(self):
        self.config = '--oem 3 --psm 3'



    def extract_text(self, image):
        gray_image = gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        #kernel = np.ones((5, 5), np.uint8)
        #denoised_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        image_height, image_width, _ = image.shape

        aspect_ratio = image_width / image_height


        if aspect_ratio > 2:
            self.config = '--oem 3 --psm 6'


        text_regions = []
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_image)

        """Group connected components based on Y-coordinates (heuristic approach.)"""
        section_groups = {}
        section_threshold = 20
        for i in range(1, retval):
            x, y, w, h, area = stats[i]
            section_groups.setdefault(y // section_threshold, []).append((x, y, w, h))

        for section_key, components in section_groups.items():
            section_image =gray_image[components[0][1]:components[-1][1] + components[-1][3], :]
            extracted_text, section_boxes = self._extract_text_and_boxes(section_image)
            text_regions.append((extracted_text, section_boxes))

        return text_regions

    def _extract_text_and_boxes(self, component):
        extracted_text = pytesseract.image_to_string(component, config=self.config)
        boxes = self.get_text_boxes(component)
        return extracted_text, boxes

    def get_text_boxes(self, image):
        results = pytesseract.image_to_data(image, output_type=Output.DICT)
        image_with_boxes = image.copy()
        boxes = []
        for i in range(0, len(results["text"])):
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]
            text = results["text"][i]
            conf = float(results["conf"][i])
            if conf > 0:
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img=image_with_boxes, text=text, org=(x, y), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=0.3, color=(36, 0, 255), thickness=1)
                boxes.append((x, y, w, h))

        return image_with_boxes, boxes

    def get_boxes(self, image):
        results = pytesseract.image_to_data(image, output_type=Output.DICT)
        boxes = []
        for i in range(0, len(results["text"])):
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]
            # extract the OCR text itself along with the confidence of the
            # text localization
            text = results["text"][i]
            conf = float(results["conf"][i])
            if conf > 0:
                boxes.append((x, y, w, h))

        return boxes


