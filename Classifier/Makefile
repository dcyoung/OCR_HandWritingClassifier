# PROJECT NAME
NAME = ocr

# SOURCE TYPE (EXPECTED: .c OR .cpp)
SRCTYPE = .cpp

MATLAB_ROOT=/Applications/MATLAB_R2015a.app/bin/maci64
OPENCV_ROOT=/Applications/opencv-2.4.10/modules

# DIRECTORIES: SOURCE, INCLUDE, OBJECT
SDIR = ./src/src
IDIR = ./src/incl -I/Applications/MATLAB_R2015a.app/extern/include -I/$(OPENCV_ROOT)/core/include -I/$(OPENCV_ROOT)/highgui/include -I/$(OPENCV_ROOT)/imgproc/include -I/$(OPENCV_ROOT)/flann/include
ODIR = ./obj

# LIBRARIES (DEFAULT: none)
LIBS = `pkg-config --libs opencv` $(MATLAB_ROOT)/libmat.dylib $(MATLAB_ROOT)/libmx.dylib

CFLAGS = -I $(IDIR) -Wl,-rpath,$(MATLAB_ROOT)
ifeq ($(SRCTYPE),.c)
CC = gcc
else ifeq ($(SRCTYPE),.cpp)
CC = g++
CFLAGS += -std=c++11
else
CC = UNKNOWN
endif

SRC = $(wildcard $(SDIR)/*$(SRCTYPE))
SRC += $(wildcard $(SDIR)/*/*$(SRCTYPE))
DEPS = $(wildcard $(IDIR)/*.h)
DEPS += $(wildcard $(IDIR)/*/*.h)
OBJ = $(addprefix $(ODIR)/,$(notdir $(SRC:$(SRCTYPE)=.o)))

all: $(NAME)

$(ODIR)/%.o: $(SDIR)/%$(SRCTYPE) $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: $(SDIR)/*/%$(SRCTYPE) $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(NAME): $(OBJ)
ifeq ($(LIBS),none)
	$(CC) -o $@ $^ $(CFLAGS)
else
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)
endif

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ $(IDIR)/*~
