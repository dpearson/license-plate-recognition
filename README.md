## Utilities for Automatic License Plate Recognition ##

This repository contains various components for an automatic license plate recognition system. Currently, it isn't particularly good, but it'll hopefully get much better in the coming weeks.

This project was originally a term project for Dr. Harry Wechsler's CS482 Computer Vision class at George Mason University.

### Prerequisites ###

* OpenCV
* Tesseract
* A C++ compiler

### Building ###

	cd src
	make

### Usage ###

#### Training the classifier ####

	cd src
	../bin/train

#### Recognizing a license plate ####

	cd src
	../bin/recognize PATH_TO_LICENSE_PLATE_IMAGE

### License ###

The following third-party image datasets were used:

* [Cars 1999 (Rear) 2](http://www.vision.caltech.edu/html-files/archive.html)

   The following README information was included:

       Car dataset taken by Markus Weber.
       California Institute of Technology PhD student under Pietro Perona.

       126 images of cars from the rear. Approximate scale normalisation. Jpeg
       format.

       Taken in the Caltech parking lots. 896 x 592 jpg format

       ---------
       R. Fergus 10/4/03

All other code is Copyright 2014 David Pearson; all rights are reserved.