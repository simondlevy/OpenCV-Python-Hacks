# Makefile for lkdemo
#
# Copyright (C) 2018 Simon D. Levy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as 
# published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# Requires: OpenCV

all: lkdemo lkdemo2

run: lkdemo
	./lkdemo

lkdemo: lkdemo.o 
	g++  -g -o lkdemo  lkdemo.o `pkg-config opencv --libs`

lkdemo.o: lkdemo.cpp 
	g++  -Wall -c lkdemo.cpp  `pkg-config opencv --cflags`

lkdemo2: lkdemo2.o 
	g++  -g -o lkdemo2  lkdemo2.o `pkg-config opencv --libs`

lkdemo2.o: lkdemo2.cpp 
	g++  -Wall -c lkdemo2.cpp  `pkg-config opencv --cflags`

clean:
	rm -rf lkdemo lkdemo2 *.o *~ 
