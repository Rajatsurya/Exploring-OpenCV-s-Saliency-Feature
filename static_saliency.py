import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# load the input image
image = cv2.imread(args["image"])

# initialize OpenCV's static saliency spectral residual detector and compute the saliency map
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliencyMap) = saliency.computeSaliency(image)
saliencyMap = (saliencyMap * 255).astype("uint8")

# show the original image and saliency map from spectral residual
cv2.imshow("Original Image", image)
cv2.imshow("Spectral Residual Saliency", saliencyMap)

# initialize OpenCV's static fine-grained saliency detector and compute the saliency map
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)

# if we would like a *binary* map that we could process for contours,
# compute convex hull's, extract bounding boxes, etc., we can additionally threshold the saliency map
threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# show the fine-grained saliency map and thresholded version
cv2.imshow("Fine Grained Saliency", saliencyMap)
cv2.imshow("Thresholded Saliency", threshMap)

cv2.waitKey(0)
