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

				vector<Rect> candidate_regions = find_candidate_regions(gray);

				for (int i = 0; i < candidate_regions.size(); i++) {
					Rect region = candidate_regions.at(i);
					for (int y = MAX(region.y - 64, 0); y < MIN(region.y + region.height, gray.rows - 64); y += 5) {
						for (int x = MAX(region.x - 128, 0); x < MIN(region.x + region.width, gray.cols - 128); x += 5) {
							// Grab the window
							Mat window = gray(Rect(x, y, 128, 64));

							// Then calculate a HOG vector for use as a feature vector
							train_data.push_back(calcHOG(&window, 8, 8));

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
				/*// Then resize
				resize(gray, gray, Size(gray.cols * 2.35, gray.rows * 2.35));

				// Blur the image...
				Mat blurred(img.cols, img.rows, img.type());
				GaussianBlur(gray, blurred, Size(11, 11), 0, 0, BORDER_DEFAULT);

				// Before finding edges
				Mat edges(img.cols, img.rows, img.type());
				Canny(blurred, edges, 30, 100);

				// Find contours
				vector<vector<Point> > contours;
				vector<Point> points;
				findContours(edges, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

				// Loop through all of the contours
				for (int i = 0; i < contours.size(); i++) {
					// Approximate a polygonal curve for the contour
					vector<Point> contour = contours[i];
					approxPolyDP(contour, points, arcLength(contour, true) * 0.01, true);

					// A license plate should have 4 corner points, but we'll settle for 4-6 points
					if (points.size() < 4 || points.size() > 6) {
						continue;
					}

					// Define coordinates for the bounding box
					int x_min = gray.cols;
					int x_max = 0;
					int y_min = gray.rows;
					int y_max = 0;

					// Find the bounding box
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

					// Perform statistical classification based on the HOG
					int num_tested = 0;
					int num_pos = 0;
					for (int y = MAX(y_min - 64, 0); y < MIN(y_max, gray.rows - 64); y += 5) {
						for (int x = MAX(x_min - 128, 0); x < MIN(x_max, gray.cols - 128); x += 5) {
							// Grab the window
							Mat window = gray(Rect(x, y, 128, 64));

							// Then calculate a HOG vector for use as a feature vector
							train_data.push_back(calcHOG(&window, 8, 8));

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
				}*/
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