import cv2
from matplotlib import pyplot as plt
from services.easyocrExtract import EasyocrExtract
from services.pytesseractExtract import PytesseractExtract
def main():
    img = ("../images/invoice.png")
    extraction_method = input("Choose extraction method (1 for EasyOCR, 2 for Pytesseract): ")

    if extraction_method == "1":
        extractor = EasyocrExtract()

        show_boxes = input("Show bounding boxes? (y/n): ").lower() == 'y'
        if show_boxes:
            extractor.draw_ocr_boxes(img, "Extracted_Text")
            plt.show()
        else:
            pass
        print("----EXTRECTED TEXT----")
        extractor.extract_text(img)

    elif extraction_method == "2":
        img1= cv2.imread(img)
        show_boxes = input("Show bounding boxes? (y/n): ").lower() == 'y'
        text_extractor = PytesseractExtract()
        image_with_boxes, boxes = text_extractor.get_text_boxes(img1)
        text_regions = text_extractor.extract_text(img1)
        extracted_text = text_regions[0][0]
        boxes = text_regions[0][1]
        if show_boxes:
            cv2.imshow('Extracted Text with Bounding Boxes', image_with_boxes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No text regions detected.")

        extracted_text = text_regions[0][0]

        print("Extracted Text:", extracted_text)
    else:
        print("Invalid extraction method selected.")
        return
if __name__ == "__main__":
    main()

