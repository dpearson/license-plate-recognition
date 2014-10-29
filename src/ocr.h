//
//  ocr.h
//  license-plate-recognition
//
//  Created by David Pearson on 10/22/14.
//  Copyright (c) 2014 David Pearson. All rights reserved.
//

#ifndef __LPR__ocr
#define __LPR__ocr

#include <opencv2/opencv.hpp>

extern char *get_plate_text(cv::Mat *img_ptr);

#endif