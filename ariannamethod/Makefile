# Makefile — build libnotorch + libnotorch_vision shared libraries
# Supports: Linux (.so), macOS (.dylib)
# Usage: make

CC = cc
CFLAGS = -O2 -std=c11 -fPIC -Wall
LDFLAGS = -lm

UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
    EXT = .dylib
    SHARED = -dynamiclib
else
    EXT = .so
    SHARED = -shared
endif

LIB = libnotorch$(EXT)
LIB_VISION = libnotorch_vision$(EXT)

all: $(LIB) $(LIB_VISION)

$(LIB): notorch.c notorch.h
	$(CC) $(CFLAGS) $(SHARED) -o $@ notorch.c $(LDFLAGS)

$(LIB_VISION): notorch_vision_wrapper.c notorch_vision.h stb_image.h notorch.h $(LIB)
	$(CC) $(CFLAGS) $(SHARED) -o $@ notorch_vision_wrapper.c -L. -lnotorch $(LDFLAGS) -Wl,-rpath,'$$ORIGIN'

clean:
	rm -f libnotorch.so libnotorch.dylib libnotorch_vision.so libnotorch_vision.dylib

.PHONY: all clean
