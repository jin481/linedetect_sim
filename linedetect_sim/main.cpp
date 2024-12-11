#include "opencv2/opencv.hpp" // OpenCV 라이브러리를 포함
#include <iostream>             // 입출력 스트림을 위한 헤더
#include <queue>                // 큐를 사용하기 위한 헤더
#include <ctime>                // 시간 관련 함수 사용을 위한 헤더
#include <unistd.h>             // 유닉스 시스템에서 사용되는 함수들을 위한 헤더
#include <signal.h>             // 신호 처리를 위한 헤더

using namespace cv;  // OpenCV 이름공간을 사용
using namespace std; // C++ 표준 이름공간을 사용

int main() {
    // 표준 에러 출력을 /dev/null로 리디렉션하여, 로그를 숨깁니다.
    freopen("/dev/null", "w", stderr);

    // 비디오 파일을 열고 스트리밍 처리
    VideoCapture source("/home/jetson/linedetect_sim/videos/5_lt_cw_100rpm_out.mp4");
    if (!source.isOpened()) { // 영상이 열리지 않으면 오류 메시지를 출력하고 종료
        cout << "Camera error" << endl;
        return -1;
    }

    // 첫 번째 스트림: 컬러 영상을 전송할 스트림 설정
    string dst1 = "appsrc ! videoconvert ! video/x-raw, format=BGRx ! "
                  "nvvidconv ! nvv4l2h264enc insert-sps-pps=true ! "
                  "h264parse ! rtph264pay pt=96 ! "
                  "udpsink host=203.234.58.158 port=8001 sync=false";

    // 첫 번째 비디오 스트림을 작성 (컬러 영상)
    VideoWriter writer1(dst1, 0, 30.0, Size(640, 360), true);
    if (!writer1.isOpened()) {  // 비디오 쓰기 객체 열기 실패 시 오류 출력
        cerr << "Writer open failed!" << endl;
        return -1;
    }

    // 세 번째 스트림: 이진 영상을 전송할 스트림 설정
    string dst3 = "appsrc ! videoconvert ! video/x-raw, format=BGRx ! "
                  "nvvidconv ! nvv4l2h264enc insert-sps-pps=true ! "
                  "h264parse ! rtph264pay pt=96 ! "
                  "udpsink host=203.234.58.158 port=8003 sync=false";

    // 세 번째 비디오 스트림을 작성 (이진 영상)
    VideoWriter writer3(dst3, 0, 30.0, Size(640, 90), true);
    if (!writer3.isOpened()) {  // 비디오 쓰기 객체 열기 실패 시 오류 출력
        cerr << "Writer open failed!" << endl;
        return -1;
    }

    // 시간 측정을 위한 변수들
    struct timeval start, end1;
    Mat frame, gray, th, roi, gray2, lab1;
    TickMeter tm;  // OpenCV의 TickMeter 클래스를 사용하여 시간 측정
    Mat labels, stats, centroids; // 라벨, 통계 및 중심점을 저장하는 변수들
    double time_1, gain = 0.3; // 시간 측정 변수와 보정 값 설정
    int lvel = 0, rvel = 0, err;
    queue<double> centerq, centeryq;  // 큐 객체 선언
    centerq.push(320);  // 중앙 X 좌표 초기값 설정
    centeryq.push(45);  // 중앙 Y 좌표 초기값 설정

    // 초기에는 실제 라인 위치를 알 수 없으므로 (-1, -1)로 설정
    Point realLinePosition(-1, -1); 

    while (true) {
        // 현재 시간 측정 시작
        gettimeofday(&start, NULL);

        // 비디오 파일에서 한 프레임을 읽어옴
        source >> frame;
        if (frame.empty()) { // 프레임이 비어 있으면 오류 출력 후 종료
            cerr << "Frame empty!" << endl;
            break;
        }

        // 하단 영역을 ROI (Region of Interest)로 지정
        roi = frame(Rect(0, 270, 640, 90)); // 하단 90픽셀 크기의 영역 선택
        cvtColor(roi, gray, COLOR_BGR2GRAY);  // 컬러 영상을 그레이스케일로 변환
        gray2 = gray + (Scalar(100) - mean(gray)); // 밝기 보정
        threshold(gray2, th, 128, 255, THRESH_BINARY); // 이진화 처리

        int cnt = connectedComponentsWithStats(th, labels, stats, centroids); // 연결된 컴포넌트를 찾아 라벨, 통계, 중심점을 계산
        cvtColor(th, lab1, COLOR_GRAY2BGR); // 이진 이미지를 다시 컬러로 변환

        bool foundRealLine = false; // 실제 라인을 찾았는지 여부를 나타내는 변수

        // 각 컴포넌트에 대해 처리
        for (int i = 1; i < cnt; i++) {  // 첫 번째 컴포넌트는 배경이므로 1부터 시작
            int *p = stats.ptr<int>(i);  // 영역의 위치와 크기 정보
            double *a = centroids.ptr<double>(i);  // 영역의 중심점 좌표

            if (p[4] < 20) continue;  // 면적이 너무 작은 경우 무시

            Scalar lineColor;

            // 첫 번째로 진짜 라인을 찾을 때
            if (realLinePosition.x == -1) {
                if (abs(a[0] - 320) <= 10) { // 영상 중앙 근처라면 진짜 라인으로 설정
                    realLinePosition = Point(a[0], a[1]); // 진짜 라인 위치 설정
                    lineColor = Scalar(0, 0, 255); // 빨간색으로 표시
                    foundRealLine = true; // 진짜 라인 발견
                    // 진짜 라인의 중심점 표시
                    circle(lab1, Point(a[0], a[1]), 3, lineColor, -1); // 빨간색으로 표시
                }
            } else {
                // 진짜 라인을 이미 찾았다면, 그 라인만 추적
                if (abs(a[0] - realLinePosition.x) <= 50) { // 진짜 라인 근처
                    lineColor = Scalar(0, 0, 255); // 빨간색으로 표시
                    foundRealLine = true;
                    // 진짜 라인의 중심점과 사각형 표시
                    circle(lab1, Point(a[0], a[1]), 3, lineColor, -1);
                    rectangle(lab1, Rect(p[0] + 1, p[1] + 1, p[2] - 2, p[3] - 2), lineColor, 2);
                    // 진짜 라인의 위치 갱신
                    realLinePosition = Point(a[0], a[1]);
                }
            }
        }

        // 진짜 라인이 사라졌을 경우 마지막 위치에 중심점만 표시
        if (!foundRealLine && realLinePosition.x != -1) {
            Scalar lineColor = Scalar(0, 0, 255); // 빨간색
            // 마지막 진짜 라인 위치에 중심점만 그려줌
            circle(lab1, realLinePosition, 3, lineColor, -1); // 빨간색 중심점만
        }

        // 진짜 라인이 아닌 다른 라인들은 파란색으로 표시
        for (int i = 1; i < cnt; i++) { 
            int *p = stats.ptr<int>(i);  // 영역의 위치와 크기 정보
            double *a = centroids.ptr<double>(i);  // 영역의 중심점 좌표

            if (p[4] < 20 || abs(a[0] - realLinePosition.x) <= 50) continue; // 진짜 라인 근처 제외

            // 가짜 라인은 파란색으로 표시
            Scalar lineColor = Scalar(255, 0, 0); // 파란색으로 설정
            // 가짜 라인에는 중심점과 사각형 그리기
            circle(lab1, Point(a[0], a[1]), 3, lineColor, -1); // 파란색 중심점
            rectangle(lab1, Rect(p[0] + 1, p[1] + 1, p[2] - 2, p[3] - 2), lineColor, 2); // 사각형 크기 조정
        }

        // 실제 라인 위치와 큐에서의 차이를 계산하고 속도를 설정
        err = centerq.back() - realLinePosition.x;
        lvel = 100 - gain * err; // 왼쪽 속도 계산
        rvel = -(100 + gain * err); // 오른쪽 속도 계산

        // 시간 측정 후 출력
        gettimeofday(&end1, NULL);
        time_1 = end1.tv_sec + end1.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
        cout << "err: " << err;
        cout << ", lvel : " << lvel;
        cout <<", rvel : " << rvel;
        cout << ", time : " << time_1 << endl;

        // 컬러 영상과 이진 영상을 스트리밍
        writer1 << frame;  // 컬러 영상
        writer3 << lab1;   // 이진 영상

        // 30ms 대기
        if (waitKey(30) >= 0) { // 키 입력이 있으면 종료
            break;
        }
    }

    return 0;
}
