OBJECTS = main.o

LIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_video

all: facetracking

facetracking: $(OBJECTS)
	g++ -o facetracking $(OBJECTS) $(LIBS)

clean:
	rm -f *.o
	rm -f facetracking
