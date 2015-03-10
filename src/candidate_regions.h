//
//  recognize.cpp
//  license-plate-recognition
//
//  Created by David Pearson on 3/10/15.
//  Copyright (c) 2015 David Pearson. All rights reserved.
//

#include <opencv2/opencv.hpp>

extern bool overlaps(cv::Rect r1, cv::Rect r2);
extern cv::vector<cv::Rect> find_candidate_regions(cv::Mat img);