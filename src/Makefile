HAS_CLANG=$(shell which clang++ > /dev/null; echo $$?)
ifeq "$(HAS_CLANG)" "0"
	CC=clang++
else
	CC=g++
endif

RM=rm -rf

HAS_PKG_CONFIG=$(shell which pkg-config > /dev/null; echo $$?)
ifeq "$(HAS_PKG_CONFIG)" "0"
	CFLAGS=`pkg-config --cflags opencv`
	LFLAGS=`pkg-config --libs opencv`
else
	CFLAGS=-I/usr/local/include
	LFLAGS=-L/usr/local/lib -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab
endif

TRAIN_SRCFILES=train.cpp util.cpp hog.cpp
TRAIN_OBJFILES=$(subst .cpp,.o,$(TRAIN_SRCFILES))

TRAIN_OUTFILE=train

RECOGNIZE_SRCFILES=recognize.cpp util.cpp hog.cpp ocr.cpp
RECOGNIZE_OBJFILES=$(subst .cpp,.o,$(RECOGNIZE_SRCFILES))

RECOGNIZE_OUTFILE=recognize

BIN_DIR=../bin

all: train recognize

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

train: $(TRAIN_OBJFILES)
	$(CC) -o $(BIN_DIR)/$(TRAIN_OUTFILE) $(TRAIN_OBJFILES) $(LFLAGS)

recognize: $(RECOGNIZE_OBJFILES)
	$(CC) -o $(BIN_DIR)/$(RECOGNIZE_OUTFILE) $(RECOGNIZE_OBJFILES) $(LFLAGS)

cleanobjs:
	$(RM) $(TRAIN_OBJFILES)
	$(RM) $(RECOGNIZE_OBJFILES)

clean: cleanobjs
	$(RM) $(BIN_DIR)/$(TRAIN_OUTFILE)
	$(RM) $(BIN_DIR)/$(RECOGNIZE_OUTFILE)
