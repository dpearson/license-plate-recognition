//
//  util.cpp
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

#include "util.h"

using namespace cv;

/* Reads an image in from a file.
 *
 * folder - The full or relative path of the folder from
 *          which to read the image
 * fname - The image's filename
 *
 * Returns the loaded image.
 */
Mat read_img(const char *folder, const char *fname) {
	// Build the full path
	char *full_path = (char *)malloc((strlen(folder) + strlen(fname) + 2) * sizeof(char));
	strcpy(full_path, folder);
	strcat(full_path, "/");
	strcat(full_path, fname);

	// Read in the image
	Mat img = imread(full_path);

	// Clean up
	free(full_path);

	return img;
}

annotation *annotation_load(char *image_name) {
	const char *dir_path = "../train_data/annotations";
	char *full_path = (char *)malloc((strlen(image_name) + strlen(dir_path) + 6) * sizeof(char));
	strcpy(full_path, dir_path);
	strcat(full_path, "/");
	strcat(full_path, image_name);
	strcat(full_path, ".txt");

	FILE *f = fopen(full_path, "r");
	free(full_path);

	annotation *an = (annotation *)malloc(sizeof(annotation));
	an->plate_number = (char *)malloc(10 * sizeof(char));

	fscanf(f, "%d %d\n%d %d\n%d %d\n%d %d\n%s", &(an->top_left.x), &(an->top_left.y), &(an->top_right.x), &(an->top_right.y), &(an->bottom_right.x), &(an->bottom_right.y), &(an->bottom_left.x), &(an->bottom_left.y), an->plate_number);

	fclose(f);

	return an;
}

void annotation_free(annotation *an) {
	free(an->plate_number);
	free(an);
}

bool overlaps(Rect r1, Rect r2) {
	if (r1.x + r1.width < r2.x || r2.x + r2.width < r1.x) {
		return false;
	}

	if (r1.y + r1.height < r2.y || r2.y + r2.height < r2.y) {
		return false;
	}

	return true;
}