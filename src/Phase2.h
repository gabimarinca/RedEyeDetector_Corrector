#ifndef PHASE2_H
#define PHASE2_H

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

const int n8_di[8] = {0,-1,-1, -1, 0, 1, 1, 1};
const int n8_dj[8] = {1, 1, 0, -1, -1,-1, 0, 1};

const int n4_di[] = {-1, 0, 1, 0};
const int n4_dj[] = {0, 1, 0, -1};

typedef struct {
    Mat B;
    Mat G;
    Mat R;
} image_channels_bgr;

typedef struct {
    Mat H;
    Mat S;
    Mat V;
} image_channels_hsv;

// Data structures
struct edge_image_values {
    int min_value;
    int max_value;
};

struct labels {
    Mat labelsM;
    int no_labels;
};


Mat negative_image(Mat image);

Mat bgr_2_grayscale(Mat source);

int* compute_histogram_naive(Mat source);

Mat apply_histogram_equalization(Mat source, int* histogram);

Mat threshold_inverse_binary(Mat gray, int thresh_value);  //binarizeaza in functie de un threshold

labels BFS_labeling(Mat source);

Mat color_labels(labels labels_str);
vector<Rect> detectEyes(Mat source); //functia main de detectie ochi

Mat displayEyeZones(Mat img, vector<Rect> eyes);

image_channels_bgr break_channels(Mat source);

image_channels_hsv bgr_2_hsv(image_channels_bgr bgr_channels);

image_channels_bgr fixRedEyes(image_channels_bgr bgr_channels, image_channels_hsv hsv_channels, vector<Rect> eyes);

#endif // PHASE2_H