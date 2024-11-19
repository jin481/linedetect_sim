#ifndef VISION_HPP
#define VISION_HPP

#include <opencv2/opencv.hpp>
using namespace cv;

class Vision {
public:
    Vision();
    ~Vision();
    void detectLine(const Mat& frame, Mat& output);
};

#endif
