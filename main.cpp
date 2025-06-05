#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/Phase2.h"

using namespace std;
using namespace cv;

int main() {
    Mat source = imread("D:\\AA-UTCN\\An III\\Sem II\\PI\\Project\\Phase2 - Implementation\\images\\face1.bmp",
                        IMREAD_COLOR);
    // original image
    imshow("Original", source);
    Mat detectedEyes = source.clone();

    vector<Rect> eyes = detectEyes(source);
    displayEyeZones(detectedEyes, eyes);
    imshow("Detected eyes", detectedEyes);

    Mat result = source.clone();
    image_channels_bgr bgr = break_channels(result);
    image_channels_hsv hsv = bgr_2_hsv(bgr);

    bgr = fixRedEyes(bgr, hsv, eyes);

    for (int i = 0; i < bgr.B.rows; i++) {
        for (int j = 0; j < bgr.B.cols; j++) {
            uchar b = bgr.B.at<uchar>(i, j);
            uchar g = bgr.G.at<uchar>(i, j);
            uchar r = bgr.R.at<uchar>(i, j);
            result.at<Vec3b>(i, j) = Vec3b(b, g, r);
        }
    }

    imshow("Fixed Image", result);

    Mat detectedEyes2 = source.clone();
    vector<Rect> eyes2 = detectEyesWithHaarCascade(result);  //advanced detection
    displayEyeZones(detectedEyes2, eyes2);
    imshow("Detected eyes with HaarCascade", detectedEyes2);

    waitKey(0);

    return 0;
}