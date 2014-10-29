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

SRCFILES=annotate.cpp
OBJFILES=$(subst .cpp,.o,$(SRCFILES))

OUTFILE=annotate

all: bin

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

bin: $(OBJFILES)
	$(CC) -o $(OUTFILE) $(OBJFILES) $(LFLAGS)

cleanobjs:
	$(RM) $(OBJFILES)

clean: cleanobjs
	$(RM) $(OUTFILE)
