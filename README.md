# linedetect_sim

## 라인검출

### 개요
이 프로그램은 비디오 파일을 분석하고 관심 영역(ROI)을 추출하여 라인을 감지합니다. 감지된 라인의 위치는 프레임 중심과 가까운 주요 라인을 식별하고 원본 및 처리된 영상을 GStreamer를 통해 스트리밍합니다.

### 코드 설명
#### main.cpp
1. ##### 초기화
    * 표준 에러 출력을 `/dev/null`로 리다이렉트하여 로그를 숨깁니다.
    * OpenCV의 `VideoCapture`를 사용하여 비디오 파일을 엽니다. 파일을 열 수 없는 경우 프로그램이 종료됩니다.

2. #### GStreamer 파이프라인
    * 스트리밍을 위한 두 개의 GStreamer 파이프라인이 정의되어 있습니다:
        * `dst1`: 원본 비디오 프레임을 전송합니다.
        * `dst3`: 라인 감지 결과를 보여주는 이진화된 프레임을 전송합니다.

3. #### 라인 감지 및 처리
    * 각 프레임의 하단 90픽셀 영역이 ROI로 사용됩니다.
    * ROI는 그레이스케일로 변환되고 밝기가 조정됩니다.
    * 이진화 처리로 라인과 유사한 구조를 분리합니다.
    * 연결된 구성 요소 분석을 통해 프레임 중심과 가장 가까운 라인을 식별하고 추적합니다.

4. #### 모터 속도 계산
    * 감지된 라인과 프레임 중심 간의 x축 좌표 차이를 바탕으로 왼쪽 및 오른쪽 모터 속도(`lvel` 및 `rvel`)를 계산합니다.

5. #### 비디오 스트리밍
    * 원본 비디오(`writer1`)와 처리된 비디오(`writer3`) 모두 정의된 GStreamer 파이프라인을 사용하여 스트리밍됩니다.

6. #### 성능 모니터링
    * 각 프레임 처리 시간은 측정되어 출력됩니다.

### Makefile
`Makefile`은 프로그램을 컴파일하는 데 사용됩니다. 주요 구성 요소:
* `CX`: C++ 컴파일러를 지정합니다.
* `CXFLAGS`: 디버깅 및 에러 보고를 위한 플래그입니다.
* `CVFLAGS`: pkg-config를 사용하여 OpenCV를 포함하고 링크합니다.
* `SRCS`: 소스 파일(main.cpp)을 지정합니다.
* `TARGET`: 실행 파일의 이름(camera)을 정의합니다.

### 출력 설명
* `콘솔 출력`: 실시간으로 처리 정보를 출력하며, 여기에는 에러 값(**err**), 계산된 모터 속도(**lvel** 및 **rvel**), 프레임당 처리 시간이 포함됩니다.
* `비디오`:
    * dst1: 원본 비디오 프레임.
    * dst3: 라인 감지 결과가 주석으로 표시된 이진화된 프레임.



https://github.com/user-attachments/assets/2d10150b-d35e-47f8-921f-5f3e2558022d


https://youtu.be/3Mr2VscjbLw




https://github.com/user-attachments/assets/04e472f5-6e9b-4e57-ba3d-dfb37c7990d2


https://youtu.be/a9swp05I8EQ


## 모션제어 결과 동영상


https://youtu.be/y11rsXyF5Z8

