//
//  recognize.cpp
//  license-plate-recognition
//
//  Created by David Pearson on 3/10/15.
//  Copyright (c) 2015 David Pearson. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "util.h"

using namespace cv;

Mat marked;

/* Find the length of the line segment containing the point (x, y).
 *
 * edge_img - A pointer to the binary image created from the Sobel result
 * x - The x coordinate of the point
 * y - The y coordinate of the point
 * mark - A flag indicating whether to mark the pixels as visited
 *
 * Returns the length of the line segment.
 */
int find_length(Mat *edge_img, int x, int y, bool mark) {
	// Dereference the image pointer
	Mat edges = *edge_img;

	// Initialize the length
	int length = 0;

	// And store the y coordinate
	int y_orig = y;

	// Iterate down the line segment
	while (y < edges.rows) {
		// Grab a 1x3 slice of the edge image
		uint16_t left = edges.at<uint16_t>(y, x - 1, 0);
		uint16_t center = edges.at<uint16_t>(y, x, 0);
		uint16_t right = edges.at<uint16_t>(y, x + 1, 0);

		// Bail if all three pixels are zero
		if (left == 0 && center == 0 && right == 0) {
			break;
		}

		// Mark the pixels as visited if necessary
		if (mark) {
			marked.at<uint16_t>(y, x - 1, 0) = 1;
			marked.at<uint16_t>(y, x, 0) = 1;
			marked.at<uint16_t>(y, x + 1, 0) = 1;
		}

		// Increment the length and y coordinate
		length++;
		y++;
	}

	// Return to the original point
	y = y_orig - 1;

	// And go up instead
	while (y >= 0) {
		// Grab a 1x3 slice of the edge image
		uint16_t left = edges.at<uint16_t>(y, x - 1, 0);
		uint16_t center = edges.at<uint16_t>(y, x, 0);
		uint16_t right = edges.at<uint16_t>(y, x + 1, 0);

		// Bail if all three pixels are zero
		if (left == 0 && center == 0 && right == 0) {
			break;
		}

		// Mark the pixels as visited if necessary
		if (mark) {
			edges.at<uint16_t>(y, x - 1, 0) = 0;
			edges.at<uint16_t>(y, x, 0) = 0;
			edges.at<uint16_t>(y, x + 1, 0) = 0;
		}

		// Increment the length and decrement the y coordinate
		length++;
		y--;
	}

	return length;
}

/* Find the x coordinate and length of the line segment corresponding
 * to the line segment containing the point (x, y).
 *
 * edge_img - A pointer to the binary image created from the Sobel result
 * x - The x coordinate of the point
 * y - The y coordinate of the point
 * parallel_length - A pointer to an integer to store the line length in
 *
 * Returns the x coordinate of the line segment.
 */
int find_corresponding_line(Mat *edge_img, int x, int y, int *parallel_length) {
	// Dereference
	Mat edges = *edge_img;

	// Find the length of the current line segment
	int length = find_length(&edges, x, y, true);

	// Throw out areas that are too small or big
	if (length < 25 || length > 75) {
		return -1;
	}

	// License plates have a ratio of 2:1, so set our search space accordingly
	double start = x + floor(length / 0.6);
	double end = x + ceil(length / 0.4);

	// Iterate through the search space
	for (int x_search = (int)start; x_search < min((int)end, edges.cols); x_search++) {
		// Find the length of the line through this point
		int len_line = find_length(&edges, x_search, y + (length / 2), false);

		// Skip this point if the length is zero
		if (len_line <= 0) {
			continue;
		}

		// Accept this line only if it's reasonably close in length to the other line
		if (abs(len_line - length) < (double)length * 0.5) {
			// Set the line length and return
			*parallel_length = max(len_line, length);
			return x_search;
		}
	}

	// Return a negative coordinate to indicate that no line was found
	return -1;
}

vector<Rect> find_candidate_regions(Mat gray) {
	// Blur the image...
	Mat blurred(gray.cols, gray.rows, gray.type());
	GaussianBlur(gray, blurred, Size(11, 11), 0, 0, BORDER_DEFAULT);

	// Before finding edges
	Mat edges_x, edges_y;
	Mat bin_img_x, bin_img_y;
	Mat edges(gray.rows, gray.cols, CV_8UC1);

	Sobel(gray, edges_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	threshold(edges_x, bin_img_x, 20, 255, CV_THRESH_BINARY);

	Mat struct_elem = getStructuringElement(MORPH_RECT, Size(2, 20));
	morphologyEx(bin_img_x, bin_img_x, MORPH_OPEN, struct_elem, Point(-1, -1), 1);

	#ifdef DEBUG
		imwrite("edges_x.png", bin_img_x);
	#endif

	marked = Mat::zeros(gray.rows, gray.cols, CV_8UC1);

	vector<Rect> regions;

	for (int y = 0; y < gray.rows; y++) {
		for (int x = 0; x < gray.cols; x++) {
			uint16_t val_x = bin_img_x.at<uint16_t>(y, x, 0);
			uint8_t is_marked = marked.at<uint8_t>(y, x, 0);

			if (val_x == 0 || is_marked != 0) {
				continue;
			}

			int length = 0;
			int x_parallel = find_corresponding_line(&bin_img_x, x, y, &length);
			if (x_parallel > 0) {
				Rect region = Rect(x, y, x_parallel - x, length);
				regions.push_back(region);
			}
		}
	}

	#ifdef DEBUG
		printf("Found %lu candidate regions\n", regions.size());
	#endif

	vector<Rect> reduced_regions;
	uint8_t *pruned = (uint8_t *)calloc(regions.size(), sizeof(uint8_t));

	for (int i = 0; i < regions.size(); i++) {
		if (pruned[i]) {
			continue;
		}

		Rect region1 = regions.at(i);

		for (int j = i + 1; j < regions.size(); j++) {
			Rect region2 = regions.at(j);
			if (overlaps(region1, region2)) {
				int area1 = region1.width * region1.height;
				int area2 = region2.width * region2.height;

				if (area2 > area1) {
					region1 = region2;
				} else {
					pruned[j] = 1;
				}
			}
		}

		reduced_regions.push_back(region1);
	}

	#ifdef DEBUG
		printf("But I cut that down to %lu real candidate regions\n", reduced_regions.size());
	#endif

	for (int i = 0; i < reduced_regions.size(); i++) {
		Rect region = reduced_regions.at(i);

		#ifdef DEBUG
			try {
				namedWindow("edges");
				imshow("edges", gray(region));
				waitKey();
			} catch (...) {}
		#endif
	}

	return reduced_regions;
}