//
//  ocr.cpp
//  license-plate-recognition
//
//  Created by David Pearson on 10/22/14.
//  Copyright (c) 2014 David Pearson. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>

#include "constants.h"
#include "ocr.h"

using namespace cv;
using namespace tesseract;

char *get_plate_text(Mat *img_ptr) throw () {
	// Dereference the image pointer
	Mat img = *img_ptr;

	Mat gray = img.clone();

	// Blur the image...
	Mat blurred(gray.cols, gray.rows, gray.type());
	GaussianBlur(gray, blurred, Size(11, 11), 0, 0, BORDER_DEFAULT);

	// Before finding edges
	Mat edges_x(gray.rows, gray.cols, CV_8UC1);
	Mat edges(gray.rows, gray.cols, CV_8UC1);

	Sobel(gray, edges_x, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	threshold(edges_x, edges_x, 100, 255, CV_THRESH_BINARY);

	Mat struct_elem = getStructuringElement(MORPH_RECT, Size(1, 5));
	morphologyEx(edges_x, edges_x, MORPH_OPEN, struct_elem, Point(-1, -1), 1);

	int rmin = img.rows;
	int rmax = 0;
	int thresh = 12;
	for (int i = 0; i < img.rows; i++) {
		int count = 0;
		for (int j = 0; j < img.cols; j++) {
			int pixel = edges_x.at<uint8_t>(i, j, 0);
			if (pixel > 0) {
				count++;
			}
		}

		#ifdef DEBUG
			printf("%d: %d\n", i, count);
		#endif

		if (count >= thresh) {
			if (i < rmin) {
				rmin = i;
			}

			if (i > rmax) {
				rmax = i;
			}
		}
	}

	// Chop off the top and bottom of the image to get a better look
	// at the license plate number
	int height = std::min(rmax - rmin + 5, img.rows - rmin - 1);
	if (height >= 10) {
		img = img(Rect(5, rmin, img.cols - 10, height));
	} else {
		img = img(Rect(5, img.rows / 5, img.cols - 10, img.rows * 4 / 5));
	}

	// Perform thresholding to get a clean image for Tesseract
	threshold(img, img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	int count = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			uint8_t pixel = img.at<uint8_t>(i, j, 0);
			if (pixel == 0) {
				count--;
			} else {
				count++;
			}
		}
	}

	if (count < 0) {
		bitwise_not(img, img);
	}

	#ifdef DEBUG
		#ifdef SHOW_IMAGES
			namedWindow("plate");
			imshow("plate", img);
			waitKey();

			imwrite("plate_text.png", img);
		#endif
	#endif

	resize(img, img, Size(img.cols * 0.75, img.rows * 0.75));

	// Create and initialize a Tesseract API instance
	TessBaseAPI *t = new TessBaseAPI();
	t->Init(NULL, "lp", OEM_DEFAULT);

	// Assum that there's only one block of text in the image
	// at this point
	t->SetPageSegMode(PSM_SINGLE_BLOCK);

	// Whitelist the characters that can actually show up in license
	// plates
	t->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345789-");

	// Load our image
	t->TesseractRect(img.data, 1, img.step1(), 0, 0, img.cols, img.rows);

	// Grab the text
	char *plate_text = t->GetUTF8Text();

	// Then dispose of the Tesseract API instance
	t->End();

	// Remove all but the first line
	char *newline_ptr = strchr(plate_text, '\n');
	*newline_ptr = '\0';

	// Return the plate text
	return plate_text;
}