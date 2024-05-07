import cv2
from services.pytesseractExtract import PytesseractExtract

image_path = '../images/invoice.png'
image = cv2.imread(image_path)

def main():
    # Create TextExtractor instance
    text_extractor = PytesseractExtract()
    # Extract text and bounding boxes
    image_with_boxes, boxes = text_extractor.get_text_boxes(image)
    text_regions = text_extractor.extract_text(image)

    if boxes:
        # Display image with bounding boxes
        cv2.imshow('Extracted Text with Bounding Boxes', image_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No text regions detected.")


    # Access extracted text and bounding boxes
    extracted_text = text_regions[0][0]


    # Print extracted text
    print("Extracted Text:")
    print(extracted_text)
if __name__ == "__main__":
    main()