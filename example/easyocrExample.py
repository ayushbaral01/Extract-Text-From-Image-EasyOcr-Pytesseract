import matplotlib.pyplot as plt

from services.easyocrExtract import EasyocrExtract


def main():

  img = ("../images/invoice.png")
  extractor = EasyocrExtract()
  extractor.extract_text(img)
  extractor.draw_ocr_boxes(img, "Extracted_Text")
  plt.show()

if __name__ == "__main__":
  main()