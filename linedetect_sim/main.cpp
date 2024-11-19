#include <iostream>
#include <opencv2/opencv.hpp>
#include "vision.hpp"
#include "dxl.hpp"

using namespace std;
using namespace cv;

int main() {
    // 비전 객체 생성
    Vision vision;

    // 동영상 파일 로드
    VideoCapture cap1("videos/5_lt_cw_100rpm_out.mp4");
    VideoCapture cap2("videos/7_lt_ccw_100rpm_in.mp4");

    if (!cap1.isOpened() || !cap2.isOpened()) {
        cerr << "Error: Cannot open video files!" << endl;
        return -1;
    }

    // Dynamixel 초기화
    Dxl dxl;
    if (!dxl.open()) {
        cerr << "Error: Unable to open Dynamixel!" << endl;
        return -1;
    }

    Mat frame;
    while (true) {
        // 첫 번째 동영상에서 프레임 읽기
        cap1 >> frame;
        if (frame.empty()) break;

        // 라인 검출
        Mat output;
        vision.detectLine(frame, output);

        // 결과 출력
        imshow("Line Detection", output);
        
        // 'q'를 누르면 종료
        if (waitKey(10) == 'q') break;
    }

    // 두 번째 동영상에서 비슷한 작업을 반복
    while (true) {
        cap2 >> frame;
        if (frame.empty()) break;

        // 라인 검출
        Mat output;
        vision.detectLine(frame, output);

        // 결과 출력
        imshow("Line Detection", output);
        
        // 'q'를 누르면 종료
        if (waitKey(10) == 'q') break;
    }

    dxl.close(); // Dynamixel 종료
    return 0;
}
