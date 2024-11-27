#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    VideoCapture source("/home/jetson/linedetect_sim/videos/7_lt_ccw_100rpm_in.mp4");
    if (!source.isOpened()) {
        cout << "Camera error" << endl;
        return -1;
    }

    // 첫 번째 스트림 (컬러 영상)
    string dst1 = "appsrc ! videoconvert ! video/x-raw, format=BGRx ! "
                  "nvvidconv ! nvv4l2h264enc insert-sps-pps=true ! "
                  "h264parse ! rtph264pay pt=96 ! "
                  "udpsink host=203.234.58.158 port=8001 sync=false";

    VideoWriter writer1(dst1, 0, 30.0, Size(640, 360), true);
    if (!writer1.isOpened()) {
        cerr << "Writer open failed!" << endl;
        return -1;
    }

    // 세 번째 스트림 (이진 영상)
    string dst3 = "appsrc ! videoconvert ! video/x-raw, format=BGRx ! "
                  "nvvidconv ! nvv4l2h264enc insert-sps-pps=true ! "
                  "h264parse ! rtph264pay pt=96 ! "
                  "udpsink host=203.234.58.158 port=8003 sync=false";

    VideoWriter writer3(dst3, 0, 30.0, Size(640, 90), true);
    if (!writer3.isOpened()) {
        cerr << "Writer open failed!" << endl;
        return -1;
    }

    Mat frame, gray, th, roi, gray2, lab1;
    TickMeter tm;  // OpenCV TickMeter 클래스 사용
    Mat labels, stats, centroids;

    // 중간 라인 위치를 계산 (수평 방향)
    int middle = 320;  // 640x360 영상에서 수평 중앙
    int left_half = 320;  // 왼쪽 절반 영역 (x < 320)

    // 진짜 라인의 마지막 위치 추적
    Point lastRealLinePosition(-1, -1); // (x, y) 좌표로 초기화

    while (true) {
        tm.start();  // 실행 시간 측정 시작

        source >> frame;
        if (frame.empty()) {
            cerr << "Frame empty!" << endl;
            break;
        }

        roi = frame(Rect(0, 270, 640, 90)); // 하단 영역 ROI
        cvtColor(roi, gray, COLOR_BGR2GRAY);
        gray2 = gray + (Scalar(100) - mean(gray));
        threshold(gray2, th, 128, 255, THRESH_BINARY);

        int cnt = connectedComponentsWithStats(th, labels, stats, centroids);
        cvtColor(th, lab1, COLOR_GRAY2BGR);

        bool foundRealLine = false;

        // 라인 후보 영역에 대해 진짜 라인과 가짜 라인 구별
        for (int i = 1; i < cnt; i++) {
            int *p = stats.ptr<int>(i); // 영역의 위치 및 크기
            double *a = centroids.ptr<double>(i); // 영역의 중심점

            if (p[4] < 20) continue; // 너무 작은 영역은 무시

            // 영역의 중심이 영상의 왼쪽 절반에 있으면 "진짜 라인"으로 간주
            Scalar lineColor;
            if (a[0] < left_half) {
                // 왼쪽 절반에 있는 라인은 진짜 라인 (빨간색으로 표시)
                lineColor = Scalar(0, 0, 255);
                lastRealLinePosition = Point(a[0], a[1]); // 마지막 위치 갱신
                foundRealLine = true;
            } else if (abs(a[0] - middle) < 50) {
                // 중앙 근처의 라인도 진짜 라인 (빨간색으로 표시)
                lineColor = Scalar(0, 0, 255);
                lastRealLinePosition = Point(a[0], a[1]); // 마지막 위치 갱신
                foundRealLine = true;
            } else {
                // 나머지 라인 (파란색으로 표시)
                lineColor = Scalar(255, 0, 0);
            }

            // 사각형과 중심점 그리기
            rectangle(lab1, Rect(p[0], p[1], p[2], p[3]), lineColor, 2);
            circle(lab1, Point(a[0], a[1]), 2, lineColor, 2);
        }

        // 진짜 라인이 사라졌을 경우 마지막 위치에 표시
        if (!foundRealLine && lastRealLinePosition.x != -1) {
            Scalar lineColor = Scalar(0, 0, 255); // 빨간색
            // 마지막 진짜 라인 위치에 원을 그려줌
            circle(lab1, lastRealLinePosition, 2, lineColor, 2);
        }

        writer1 << frame;  // 컬러 영상
        writer3 << lab1;     // 이진 영상

        tm.stop();  // 실행 시간 측정 종료
        cout << "Frame processing time: " << tm.getTimeMilli() << " ms" << endl;
        tm.reset();  // 실행 시간 초기화

        // 30ms 대기
        if (waitKey(30) >= 0) {
            break;
        }
    }

    return 0;
}
