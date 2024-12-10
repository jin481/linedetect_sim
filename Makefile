CX = g++   
CXFLAGS = -g -Wall
CVFLAGS = `pkg-config --cflags --libs opencv4`

SRCS = main.cpp      
TARGET = camera

$(TARGET): $(SRCS)
	$(CX) $(CXFLAGS) -o $(TARGET) $(SRCS) $(CVFLAGS)

.PHONY: clean
clean:
		rm -f $(TARGET)