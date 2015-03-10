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

	// Chop off the top and bottom of the image to get a better look
	// at the license plate number
	// TODO: FIXME
	img = img(Rect(0, img.rows / 4, img.cols, img.rows / 2));
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