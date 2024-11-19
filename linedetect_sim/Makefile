CXX = g++
CXXFLAGS = -std=c++11 -g
LDFLAGS = `pkg-config --cflags --libs opencv4`
SOURCES = main.cpp vision.cpp dxl.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = linedetect_sim

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(EXECUTABLE) $(LDFLAGS)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
