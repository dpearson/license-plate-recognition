//
//  ocr.cpp
//  license-plate-recognition
//
//  Created by David Pearson on 10/22/14.
//  Copyright (c) 2014 David Pearson. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>

#include "ocr.h"

using namespace cv;
using namespace tesseract;

char *get_plate_text(Mat *img_ptr) {
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

	int min = img.rows;
	int max = 0;
	int thresh = 10;
	for (int i = 0; i < img.rows; i++) {
		int count = 0;
		for (int j = 0; j < img.cols; j++) {
			int pixel = edges_x.at<uint8_t>(i, j, 0);
			if (pixel > 0) {
				count++;
			}
		}

		if (count >= thresh) {
			if (i < min) {
				min = i;
			}

			if (i > max) {
				max = i;
			}
		}
	}

	//printf("%d, %d\n", min, max);

	// Chop off the top and bottom of the image to get a better look
	// at the license plate number
	img = img(Rect(5, min, img.cols - 10, max - min));
	/*namedWindow("plate");
	imshow("plate", img);
	waitKey();*/

	// Perform thresholding to get a clean image for Tesseract
	threshold(img, img, 150, 255, CV_THRESH_BINARY);

	// Create and initialize a Tesseract API instance
	TessBaseAPI *t = new TessBaseAPI();
	t->Init(NULL, "eng", OEM_DEFAULT);

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
	printf("%s\n", plate_text);

	// Then dispose of the Tesseract API instance
	t->End();

	// Return the plate text
	return plate_text;
}