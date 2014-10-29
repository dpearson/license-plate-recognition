//
//  hog.cpp
//  license-plate-recognition
//
//  Created by David Pearson on 10/22/14.
//  Copyright (c) 2014 David Pearson. All rights reserved.
//

#include <opencv2/opencv.hpp>

#include "hog.h"

using namespace cv;

Mat calcHOG(Mat *img_ptr, int num_cells, int num_bins) {
	// Dereference
	Mat img = *img_ptr;

	// Calculate the derivatives in the X and Y directions
	Mat sobel_x(img.cols, img.rows, img.type());
	Mat sobel_y(img.cols, img.rows, img.type());
	Sobel(img, sobel_x, CV_32F, 1, 0);
	Sobel(img, sobel_y, CV_32F, 0, 1);

	// Then threshold the result
	threshold(sobel_x, sobel_x, 127, 255, THRESH_BINARY);
	threshold(sobel_y, sobel_y, 127, 255, THRESH_BINARY);

	// Initialize an empty vector
	Mat hog = Mat::zeros(1, num_cells * num_cells * num_bins, CV_32F);

	// Calculate cell and bin sizes
	int cell_size_x = img.cols / num_cells;
	int cell_size_y = img.rows / num_cells;
	float bin_size_angle = M_PI / num_bins;

	for (int y = 0; y < num_cells; y++) {
		for (int x = 0; x < num_cells; x++) {
			for (int i = 0; i < cell_size_y; i++) {
				for (int j = 0; j < cell_size_x; j++) {
					// Calculate the gradiant and its angle
					float grad_x = sobel_x.at<float>(y * cell_size_y + i, x * cell_size_x + j);
					float grad_y = sobel_y.at<float>(y * cell_size_y + i, x * cell_size_x + j);
					float grad = sqrtf(powf(grad_x, 2) + powf(grad_y, 2));
					float angle = atan2f(y * cell_size_y + i, x * cell_size_x + j);
					if (angle < 0) {
						angle = angle + 2 * M_PI;
					}

					// Calculate and set the value
					int bin = (int)floorf(angle / bin_size_angle);
					int index = num_bins * (y * num_cells + x) + bin;
					hog.at<float>(0, index, 0) += grad / (img.rows * img.cols);
				}
			}
		}
	}

	return hog;
}