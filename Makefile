OBJECTS = main.o

LIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_video

all: fmirror

fmirror: $(OBJECTS)
	g++ -o fmirror $(OBJECTS) $(LIBS)

clean:
	rm -f *.o
	rm -f fmirror
