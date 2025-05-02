First, I collected images and used Roboflow to annotate them into 4 classes: number plate, rider, with helmet, and without helmet. After annotation, I applied data augmentation to improve model generalization, and then trained a YOLOv8 model on this dataset. 
Once the training was complete, I evaluated the model using validation metrics such as mAP and precision-recall scores.

To read the number plates of helmetless riders, I integrated PaddleOCR. To improve OCR accuracy, I implemented several enhancements:

Applied preprocessing steps such as grayscale conversion, denoising, and Otsu thresholding to clean and enhance the cropped number plate region.

Added padding around number plate crops to capture more context.

Filtered out small or empty plate crops to avoid invalid OCR inputs.

Performed post-processing on OCR results by removing whitespace, converting to uppercase, and correcting common misclassifications (e.g., replacing 'O' with '0').

Matched number plates strictly within the rider bounding boxes to avoid false positives.

Selected the closest number plate to the helmetless rider when multiple plates were detected.

These enhancements significantly improved the reliability of number plate recognition in real-world traffic scenarios.

