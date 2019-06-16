

# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
 
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
    help="type of preprocessing to be done")
# image path is (ap).parse_args().image
args = vars(ap.parse_args())
print(args["image"])

for filename in os.listdir(args["image"]):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"): 
        print(os.path.join(args["image"], filename))
        oiginalFileName =  filename
        # print(args["image"])
        image = cv2.imread(os.path.join(args["image"], filename))

        # load the image and compute the ratio of the old height
        # to the new height, clone it, and resize it
        ratio = image.shape[0] / 500.0
        orig = image.copy()
        image = imutils.resize(image, height = 500)
         
        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)


        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
         
        warped=orig
        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

                break


        # apply the four point transform to obtain a top-down
        # view of the original image
         
        # convert the warped image to grayscale, then threshold it
        # to give it that 'black and white' paper effect
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        T = threshold_local(warped, 11, offset = 10, method = "gaussian")
        warped = (warped > T).astype("uint8") * 255

        # check to see if we should apply thresholding to preprocess the
        # image
        if args["preprocess"] == "thresh":
            gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # make a check to see if median blurring should be done to remove
        # noise
        elif args["preprocess"] == "blur":
            gray = cv2.medianBlur(gray, 3)
         
        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, warped)

        # load the image as a PIL/Pillow image, apply OCR, and then delete
        # the temporary file
        text = pytesseract.image_to_string(Image.open(filename))
        os.remove(filename)
        # print(type(text))
         
        # show the output images
        # cv2.imshow("Original", imutils.resize(orig, height = 650))
        # cv2.imshow("output", imutils.resize(warped, height = 650))
        # cv2.imshow("Image", image)
        # cv2.imshow("after edges", edged)
        cv2.waitKey(0)
        # print(os.path.join(args["image"], filename))
        Outfile = open('output/'+oiginalFileName+'.txt','w+')
        Outfile.write(text.encode('utf8'))
        Outfile.close()
        f = open('output/'+oiginalFileName+'.txt', 'r')
        print (f.read().decode('utf8'))




