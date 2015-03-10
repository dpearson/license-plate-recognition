//
//  recognize.cpp
//  license-plate-recognition
//
//  Created by David Pearson on 10/21/14.
//  Copyright (c) 2014-2015 David Pearson. All rights reserved.
//

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "candidate_regions.h"
#include "hog.h"
#include "ocr.h"
#include "util.h"

using namespace cv;

#define DEBUG true
#define SHOW_IMAGE true

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

	// Load the already-trained classifier
	CvNormalBayesClassifier *classifier = new CvNormalBayesClassifier();
	classifier->load("plate_classifier.xml");

	// Track the best known blob
	double max_ratio = -1.0;
	Rect max_region;

	vector<Rect> candidate_regions = find_candidate_regions(img);

	for (int i = 0; i < candidate_regions.size(); i++) {
		Rect region = candidate_regions.at(i);

		// Perform statistical classification based on the HOG
		int num_tested = 0;
		int num_pos = 0;
		for (int y = MAX(region.y - 64, 0); y < MIN(region.y + region.height, gray.rows - 64); y += 5) {
			for (int x = MAX(region.x - 128, 0); x < MIN(region.x + region.width, gray.cols - 128); x += 5) {
				// Grab the window
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

		#ifdef DEBUG
			printf("%f\n", ratio);
		#endif

		if (ratio > max_ratio) {
			max_ratio = ratio;
			max_region = region;
		}
	}

	// Isolate the best known region
	Mat plate = gray(max_region);

	printf("%f\n", ((float)plate.rows) / plate.cols);

	// Show the image if necessary
	#ifdef SHOW_IMAGE
		namedWindow("result");
		imshow("result", plate);
		waitKey();
	#endif

	// But write out the image either way
	imwrite("out_plate.png", plate);

	// OCR the plate image
	char *plate_text = get_plate_text(&plate);

	// And print out the resulting text
	printf("%s\n", plate_text);

	// Then free the plate text
	delete [] plate_text;

	return 0;
}