#include "vision.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Vision::Vision() {
    // 초기화 작업 (필요시)
}

Vision::~Vision() {
    // 리소스 해제 (필요시)
}

void Vision::detectLine(const Mat& frame, Mat& output) {
    // 그레이스케일 변환
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // 이진화 (threshold)
    Mat binary;
    threshold(gray, binary, 100, 255, THRESH_BINARY_INV); // 배경이 하얀색, 라인이 검은색

    // 에지 검출 (Canny)
    Mat edges;
    Canny(binary, edges, 100, 200);

    // 라인 검출을 위한 허프 변환
    vector<Vec2f> lines;
    HoughLines(edges, lines, 1, CV_PI / 180, 100);

    output = frame.clone();
    // 검출된 라인 그리기
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0];
        float theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(output, pt1, pt2, Scalar(0, 0, 255), 2);
    }
}
