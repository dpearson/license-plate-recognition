//
//  train.cpp
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
#include "util.h"

using namespace cv;

/* Main method
 */
int main(int argc, const char *argv[]) {
	// Folder for training images
	const char *images_folder = "../train_data/images";

	// Empty matrices for training data and responses
	Mat train_data(0, 512, CV_32F);
	Mat responses(0, 1, CV_32F);

	// Seed the random number generator for good measure
	srand(1788751783);

	// Iterate through the training images
	DIR *directory = opendir(images_folder);
	struct dirent *entry = NULL;
	while ((entry = readdir(directory)) != NULL) {
		if (entry->d_type == DT_REG) {
			// Get the filename
			char *fname = entry->d_name;

			// Ignore .DS_Store files (screw you too, Apple)
			if (strcmp(fname, ".DS_Store") == 0) {
				continue;
			}

			// Read in the training image
			Mat img = read_img(images_folder, fname);

			// Load the annotation
			annotation *an = annotation_load(fname);

			// Get the bounding coordinates of the license plate
			int left = MIN(an->top_left.x, an->bottom_left.x);
			int right = MAX(an->top_right.x, an->bottom_right.x);
			int top = MIN(an->top_left.y, an->top_right.y);
			int bottom = MAX(an->bottom_left.y, an->bottom_right.y);

			// And calculate the size of the plate
			int plate_width = right - left;
			int plate_height = bottom - top;

			Rect plate = Rect(left, top, plate_width, plate_height);

			// Ignore images that don't have license plates
			if (strcmp(an->plate_number, "N/A") != 0) {
				printf("%s\n", fname);

				// Convert the image to grayscale
				Mat gray(img.cols, img.rows, img.type());
				cvtColor(img, gray, CV_BGR2GRAY);
				imwrite("out_gray.png", gray);

				// Find candidate regions
				vector<Rect> candidate_regions = find_candidate_regions(gray);

				// Then iterate through them
				for (int i = 0; i < candidate_regions.size(); i++) {
					// Grab the current region
					Rect region = candidate_regions.at(i);

					// Perform a raster scan through the region
					for (int y = MAX(region.y - HOG_WINDOW_HEIGHT, 0); y < MIN(region.y + region.height, gray.rows - HOG_WINDOW_HEIGHT); y += 5) {
						for (int x = MAX(region.x - HOG_WINDOW_WIDTH, 0); x < MIN(region.x + region.width, gray.cols - HOG_WINDOW_WIDTH); x += 5) {
							// Grab the window
							Mat window = gray(Rect(x, y, HOG_WINDOW_WIDTH, HOG_WINDOW_HEIGHT));

							// Then calculate a HOG vector for use as a feature vector
							train_data.push_back(calcHOG(&window, HOG_CELLS, HOG_BINS));

							// Pick and set the correct response
							Mat response_row(1, 1, CV_32F);
							if (plate.contains(Point(x, y))) {
								response_row.at<float>(0, 0, 0) = 0.0;
							} else {
								response_row.at<float>(0, 0, 0) = 1.0;
							}
							responses.push_back(response_row);
						}
					}
				}
			}

			// Clean up
			annotation_free(an);
		}
	}

	// Close the directory
	closedir(directory);

	// Train and save classifier
	CvNormalBayesClassifier *classifier = new CvNormalBayesClassifier();
	classifier->train(train_data, responses);
	classifier->save("plate_classifier.xml");

	return 0;
}