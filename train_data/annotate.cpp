//
//  annotate.cpp
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

/* Handles a mouse event.
 *
 * event - The event type
 * x - The X coordinate of the event
 * y - The Y coordinate of the event
 * flags - ?
 * param - The open file to write annotation data to
 *
 * Returns nothing.
 */
static void mouse_event_cb(int event, int x, int y, int flags, void *param) {
	// Ignore non-left click events
	if (event == EVENT_LBUTTONDOWN) {
		FILE *f = (FILE *)param;

		// Write out the coordinates
		fprintf(f, "%d %d\n", x, y);
	}
}

/* Main method
 */
int main(int argc, const char *argv[]) {
	// Create our window
	namedWindow("image");

	// Define file locations
	const char *annotations_folder = "./annotations";
	const char *images_folder = "./images";

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

			// Build the annotation path
			char *full_path = (char *)malloc((strlen(annotations_folder) + strlen(fname) + 6) * sizeof(char));
			strcpy(full_path, annotations_folder);
			strcat(full_path, "/");
			strcat(full_path, fname);
			strcat(full_path, ".txt");

			// Skip the image if the annotation has already been completed
			if (access(full_path, W_OK) != -1) {
				free(full_path);
				continue;
			}

			// Then open the annotation file
			FILE *f = fopen(full_path, "w");

			free(full_path);

			// Read in the training image
			Mat img = read_img(images_folder, fname);

			// Show the image and perform setup
			imshow("image", img);
			setMouseCallback("image", mouse_event_cb, f);
			waitKey();

			// Get the plate text and write it to a file
			char *plate_text = (char *)malloc(10 * sizeof(char));
			printf("Enter plate text: ");
			scanf("%s", plate_text);
			fprintf(f, "%s", plate_text);
			free(plate_text);

			// Close the annotation file
			fclose(f);
		}
	}
	closedir(directory);

	return 0;
}