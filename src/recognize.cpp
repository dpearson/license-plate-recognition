//
//  recognize.cpp
//  license-plate-recognition
//
//  Created by David Pearson on 10/21/14.
//  Copyright (c) 2014 David Pearson. All rights reserved.
//

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "hog.h"
#include "ocr.h"
#include "util.h"

using namespace cv;

/* Main method
*/
int main(int argc, const char *argv[]) {
	if (argc != 2) {
		printf("USAGE: recognize IMAGE_FILE\n");
		return -1;
	}

	// Read in the input image
	Mat img = imread(argv[1]);

	// Convert the image to grayscale
	Mat gray(img.cols, img.rows, img.type());
	cvtColor(img, gray, CV_BGR2GRAY);
	imwrite("out_gray.png", gray);

	// Then resize
	resize(gray, gray, Size(gray.cols * 2.35, gray.rows * 2.35));

	// Blur the image...
	Mat blurred(img.cols, img.rows, img.type());
	GaussianBlur(gray, blurred, Size(11, 11), 0, 0, BORDER_DEFAULT);

	// Before finding edges
	Mat edges(img.cols, img.rows, img.type());
	Canny(blurred, edges, 30, 100);

	// Perform a bunch of morphological operations and find contours
	Mat struct_elem = getStructuringElement(MORPH_RECT, Size(21, 21));
	morphologyEx(edges, edges, MORPH_CLOSE, struct_elem, Point(-1, -1), 1);
	struct_elem = getStructuringElement(MORPH_RECT, Size(20, 20));
	erode(edges, edges, struct_elem);
	vector<vector<Point> > contours;
	findContours(edges, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	// Load the already-trained classifier
	CvNormalBayesClassifier *classifier = new CvNormalBayesClassifier();
	classifier->load("plate_classifier.xml");

	// Track the best known blob
	double max_ratio = -1.0;
	int max_x1 = -1;
	int max_x2 = -1;
	int max_y1 = -1;
	int max_y2 = -1;

	// Loop through all of the contours
	for (int i = 0; i < contours.size(); i++) {
		int x_min = gray.cols;
		int x_max = 0;
		int y_min = gray.rows;
		int y_max = 0;

		// Find the bounding box
		vector<Point> contour = contours[i];
		for (int j = 0; j < contour.size(); j++) {
			Point p = contour[j];
			x_min = MIN(p.x, x_min);
			x_max = MAX(p.x, x_max);
			y_min = MIN(p.y, y_min);
			y_max = MAX(p.y, y_max);
		}

		// Calculate the size of the bounding box
		int width = x_max - x_min;
		int height = y_max - y_min;

		// Then throw out areas that are too small
		if (width < 100 || height < 60) {
			continue;
		}

		/*namedWindow("edges");
		imshow("edges", gray(Rect(x_min, y_min, width, height)));
		waitKey();*/
		// Perform statistical classification based on the HOG
		int num_tested = 0;
		int num_pos = 0;
		for (int y = MAX(y_min - 64, 0); y < MIN(y_max, gray.rows - 64); y += 5) {
			for (int x = MAX(x_min - 128, 0); x < MIN(x_max, gray.cols - 128); x += 5) {
				Mat window = gray(Rect(x, y, 128, 64));
				float response = classifier->predict(calcHOG(&window, 8, 8));
				if (response == 0.0) {
					num_pos++;
				}
				num_tested++;
			}
		}

		// Update the best known region if necessary
		double ratio = 1.0 * num_pos / num_tested;
		printf("%f\n", ratio);
		if (ratio > max_ratio) {
			max_ratio = ratio;
			max_x1 = x_min;
			max_x2 = x_max;
			max_y1 = y_min;
			max_y2 = y_max;
		}
	}

	// Show the best known region
	Mat plate = gray(Rect(max_x1, max_y1, max_x2 - max_x1, max_y2 - max_y1));
	namedWindow("result");
	imshow("result", plate);
	waitKey();
	imwrite("out_plate.png", plate);

	// TODO: Remove me maybe?
	Mat plate_area = gray(Rect(max_x1 - 64, max_y1 - 32, max_x2 - max_x1 + 64, max_y2 - max_y1 + 64));

	// OCR the plate image
	char *plate_text = get_plate_text(&plate);

	// And print out the resulting text
	printf("%s\n", plate_text);

	// Then free the plate text
	delete [] plate_text;

	return 0;
}