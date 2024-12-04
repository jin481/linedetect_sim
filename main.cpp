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

    // 초기 진짜 라인의 위치 없음
    Point realLinePosition(-1, -1);  // (x, y) 좌표로 초기화

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

            Scalar lineColor;

            // 처음 진짜 라인을 찾을 때 (하단 중앙 320 ± 10)
            if (realLinePosition.x == -1) {
                if (abs(a[0] - 320) <= 10) { // 영상 중앙 근처
                    realLinePosition = Point(a[0], a[1]); // 진짜 라인 위치 설정
                    lineColor = Scalar(0, 0, 255); // 빨간색으로 표시
                    foundRealLine = true;
                    // 진짜 라인에는 중심점도 그리기 (작게 조정)
                    circle(lab1, Point(a[0], a[1]), 3, lineColor, -1); // 빨간색 중심점
                }
            } else {
                // 진짜 라인이 이미 설정되었으면 그 라인만 추적
                if (abs(a[0] - realLinePosition.x) <= 50) { // 진짜 라인 근처
                    lineColor = Scalar(0, 0, 255); // 빨간색으로 표시
                    foundRealLine = true;
                    // 진짜 라인에는 중심점도 그리기 (작게 조정)
                    circle(lab1, Point(a[0], a[1]), 3, lineColor, -1); // 빨간색 중심점
                    // 사각형도 그리기 (크기 작게 조정)
                    rectangle(lab1, Rect(p[0] + 2, p[1] + 2, p[2] - 4, p[3] - 4), lineColor, 2); // 사각형 크기 조정
                    // 진짜 라인의 위치 갱신
                    realLinePosition = Point(a[0], a[1]);
                }
            }
        }

        // 진짜 라인이 사라졌을 경우 마지막 위치에 중심점만 표시
        if (!foundRealLine && realLinePosition.x != -1) {
            Scalar lineColor = Scalar(0, 0, 255); // 빨간색
            // 마지막 진짜 라인 위치에 중심점만 그려줌 (작게 조정)
            circle(lab1, realLinePosition, 3, lineColor, -1); // 빨간색 중심점만
        }

        // 진짜 라인이 아닌 다른 라인들은 파란색으로 표시
        for (int i = 1; i < cnt; i++) {
            int *p = stats.ptr<int>(i); // 영역의 위치 및 크기
            double *a = centroids.ptr<double>(i); // 영역의 중심점

            if (p[4] < 20 || abs(a[0] - realLinePosition.x) <= 50) continue; // 진짜 라인 근처는 제외

            // 가짜 라인에는 파란색으로 표시
            Scalar lineColor = Scalar(255, 0, 0); // 파란색으로 설정
            // 가짜 라인에는 중심점과 사각형 그리기
            circle(lab1, Point(a[0], a[1]), 3, lineColor, -1); // 파란색 중심점
            rectangle(lab1, Rect(p[0] + 2, p[1] + 2, p[2] - 4, p[3] - 4), lineColor, 2); // 사각형 크기 조정
        }

        // 영상의 중앙 X 좌표 (로봇의 중심 x좌표)
        int centerX = frame.cols / 2;

        // 진짜 라인의 무게 중심 X 좌표
        int realLineX = realLinePosition.x;

        // 위치 오차 계산 (영상 중심 - 진짜 라인 중심)
        int error = centerX - realLineX;

        // 위치 오차 출력
        cout << "Error: " << error << endl;

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
