//
//  train.cpp
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

			// Ignore images that don't have license plates
			if (strcmp(an->plate_number, "N/A") != 0) {
				printf("%s\n", fname);

				// Convert the image to grayscale
				Mat gray(img.cols, img.rows, img.type());
				cvtColor(img, gray, CV_BGR2GRAY);
				//equalizeHist(gray, gray);

				// Get the bounding coordinates of the license plate
				int left = MIN(an->top_left.x, an->bottom_left.x);
				int right = MAX(an->top_right.x, an->bottom_right.x);
				int top = MIN(an->top_left.y, an->top_right.y);
				int bottom = MAX(an->bottom_left.y, an->bottom_right.y);

				// And calculate the size of the plate
				int width = right - left;
				int height = bottom - top;


				// Grab the license plate and scale it to a standard size
				//Mat license_plate = gray(Rect(left - 5, top - 5, width + 10, height + 10));
				//resize(license_plate, license_plate, Size(128, 64));
				Mat license_plate = gray(Rect(left, bottom - 64, 128, 64));
				int count = 0;
				for (int y = MAX(top - 32, 0); y < MIN(bottom, gray.rows - 32); y += 5) {
					for (int x = MAX(left - 64, 0); x < MIN(right, gray.cols - 64); x += 5) {
						Mat window = gray(Rect(x, y, 128, 64));
						// Calculate the HOG and add it to the training data matrix
						train_data.push_back(calcHOG(&window, 8, 8));

						// Then set the response value
						Mat response_row(1, 1, CV_32F);
						response_row.at<float>(0, 0, 0) = 0.0;
						responses.push_back(response_row);

						count++;
					}
				}

				for (int i = 0; i < count - 4; i++) {
					// Find a random area of the image to use as a negative data point
					int x = rand() % (gray.cols - 128);
					int y = rand() % (gray.rows - 64);

					while (x >= (left - 128) && x <= right) {
						x = rand() % (gray.cols - 128);
					}

					while (y >= (top - 148) && x <= bottom) {
						y = rand() % (gray.rows - 64);
					}

					// Grab the negative window and add its HOG to the training data
					Mat negative = gray(Rect(x, y, 128, 64));
					train_data.push_back(calcHOG(&negative, 8, 8));

					// Then add the proper response to the list
					Mat response_row2(1, 1, CV_32F);
					response_row2.at<float>(0, 0, 0) = 1.0;
					responses.push_back(response_row2);
				}

				// Grab the negative window and add its HOG to the training data
				Mat negative = gray(Rect(left - 128, top, 128, 64));
				train_data.push_back(calcHOG(&negative, 8, 8));

				// Then add the proper response to the list
				Mat response_row2(1, 1, CV_32F);
				response_row2.at<float>(0, 0, 0) = 1.0;
				responses.push_back(response_row2);

				// Grab the negative window and add its HOG to the training data
				Mat negative3 = gray(Rect(right, top, 128, 64));
				train_data.push_back(calcHOG(&negative3, 8, 8));

				// Then add the proper response to the list
				Mat response_row3(1, 1, CV_32F);
				response_row3.at<float>(0, 0, 0) = 1.0;
				responses.push_back(response_row3);

				// Grab the negative window and add its HOG to the training data
				Mat negative4 = gray(Rect(left, top - 64, 128, 64));
				train_data.push_back(calcHOG(&negative4, 8, 8));

				// Then add the proper response to the list
				Mat response_row4(1, 1, CV_32F);
				response_row4.at<float>(0, 0, 0) = 1.0;
				responses.push_back(response_row4);

				// Grab the negative window and add its HOG to the training data
				Mat negative5 = gray(Rect(left, bottom, 128, 64));
				train_data.push_back(calcHOG(&negative5, 8, 8));

				// Then add the proper response to the list
				Mat response_row5(1, 1, CV_32F);
				response_row5.at<float>(0, 0, 0) = 1.0;
				responses.push_back(response_row5);
			}

			// Clean up
			annotation_free(an);
		}
	}
	closedir(directory);

	// Train and save classifier
	CvNormalBayesClassifier *classifier = new CvNormalBayesClassifier();
	classifier->train(train_data, responses);
	classifier->save("plate_classifier.xml");

	return 0;
}