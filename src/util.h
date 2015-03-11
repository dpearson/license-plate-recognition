//
//  util.h
//  license-plate-recognition
//
//  Created by David Pearson on 10/21/14.
//  Copyright (c) 2014-2015 David Pearson. All rights reserved.
//

#ifndef __LPR__util
#define __LPR__util

#include <opencv2/opencv.hpp>

typedef struct {
	int x;
	int y;
} point;

typedef struct {
	point top_left;
	point top_right;
	point bottom_right;
	point bottom_left;

	char *plate_number;
} annotation;

extern annotation *annotation_load(char *image_name);
extern void annotation_free(annotation *an);

extern cv::Mat read_img(const char *folder, const char *fname);

extern bool overlaps(cv::Rect r1, cv::Rect r2);
#endif