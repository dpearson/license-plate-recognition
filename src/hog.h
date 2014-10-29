//
//  hog.h
//  license-plate-recognition
//
//  Created by David Pearson on 10/22/14.
//  Copyright (c) 2014 David Pearson. All rights reserved.
//

#ifndef __LPR__hog
#define __LPR__hog

#include <opencv2/opencv.hpp>

extern cv::Mat calcHOG(cv::Mat *img_ptr, int num_cells, int num_bins);

#endif