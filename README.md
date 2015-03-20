## Utilities for Automatic License Plate Recognition ##

This repository contains various components for an automatic license plate recognition system released under the MIT License. It currently is about 60% accurate, but, with more work, it will hopefully do much better.

This project was originally a term project for Dr. Harry Wechsler's CS482 Computer Vision class at George Mason University.

### Prerequisites ###

* OpenCV
* Tesseract
* A C++ compiler

On a recent version of Ubuntu, these dependencies can be installed by running:

	sudo apt-get install build-essential libopencv-dev libtesseract-dev

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

#### Testing the recognition utility ####

	cd src

	# Move training images into ../train_data/images
	# Move testing images into ../train_data/test_images

	../bin/train
	find ../train_data/test_images -type f -exec ../bin/recognize {} \;

### License ###

All code is made available to you under the following (MIT) license:

	The MIT License (MIT)

	Copyright (c) 2014-2015 David Pearson

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.

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