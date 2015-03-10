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
#include "constants.h"
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

	// Load the already-trained classifier
	CvNormalBayesClassifier *classifier = new CvNormalBayesClassifier();
	classifier->load("plate_classifier.xml");

	// Track the best known blob
	double max_ratio = -1.0;
	Rect max_region;

	// Find candidate regions
	vector<Rect> candidate_regions = find_candidate_regions(img);

	// Then iterate through them
	for (int i = 0; i < candidate_regions.size(); i++) {
		// Grab the current region
		Rect region = candidate_regions.at(i);

		// Perform statistical classification based on the HOG
		int num_tested = 0;
		int num_pos = 0;
		for (int y = MAX(region.y - HOG_WINDOW_HEIGHT, 0); y < MIN(region.y + region.height, gray.rows - HOG_WINDOW_HEIGHT); y += 5) {
			for (int x = MAX(region.x - HOG_WINDOW_WIDTH, 0); x < MIN(region.x + region.width, gray.cols - HOG_WINDOW_WIDTH); x += 5) {
				// Grab the window
				Mat window = gray(Rect(x, y, HOG_WINDOW_WIDTH, HOG_WINDOW_HEIGHT));

				// Predict the response for the window
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