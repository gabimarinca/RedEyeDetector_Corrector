#include "Phase2.h"
#include <cmath>
#include <random>

using namespace cv;
using namespace std;

#include <opencv2/opencv.hpp>
#include <vector>

Mat negative_image(Mat image){

    Mat negative = image.clone();

    for(int i=0; i<image.rows; i++){
        for(int j=0; j<image.cols; j++){
            negative.at<uchar>(i,j) = 255 - image.at<uchar>(i,j);
        }
    }
    return negative;
}

Mat bgr_2_grayscale(Mat source){
    int rows, cols;
    Mat grayscale_image;

    rows = source.rows;
    cols = source.cols;
    grayscale_image = Mat(rows,cols,CV_8UC1);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            unsigned int blue = source.at<Vec3b>(i,j)[0];
            unsigned int green = source.at<Vec3b>(i,j)[1];
            unsigned int red = source.at<Vec3b>(i,j)[2];
            grayscale_image.at<uchar>(i,j) = (red + green + blue) / 3;
        }
    }
    return grayscale_image;
}

int* compute_histogram_naive(Mat source){

    int* histogram = (int*)calloc(256, sizeof(int));

    int rows = source.rows;
    int cols = source.cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            histogram[source.at<uchar>(i, j)]++;
        }
    }
    return histogram;
}

Mat apply_histogram_equalization(Mat source, int* histogram){

    Mat result;
    int rows = source.rows;
    int cols = source.cols;
    result = Mat(rows, cols, CV_8UC1);
    int M = source.rows * source.cols;
    float L = 255.0f;
    float pc[256] = {0};
    pc[0] = (float)histogram[0] / M;
    for (int i = 1; i < 256; i++) {
        pc[i] = pc[i-1] + (float)histogram[i] / M;
    }
    float tab[256];
    for (int i = 0; i < 256; i++) {
        tab[i] = L*pc[i];
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            uchar g_in = source.at<uchar>(i, j);
            uchar g_out = tab[g_in];
            result.at<uchar>(i, j) = g_out;
        }
    }
    return result;
}

Mat threshold_inverse_binary(Mat gray, int thresh_value) {
    Mat binary_image = Mat(gray.rows,gray.cols, CV_8UC1);

    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            uchar pixel = gray.at<uchar>(i, j);
            binary_image.at<uchar>(i, j) = (pixel < thresh_value) ? 255 : 0;
        }
    }
    return binary_image;
}

bool IsInside(int img_rows, int img_cols, int i, int j){

    if (i<0 || i>=img_rows || j<0 || j>=img_cols)
        return false;
    return true;
}


Mat dilation(Mat source,  int no_iter){

    Mat dst, aux;
    int rows, cols;

    rows = source.rows, cols = source.cols;
    dst = source.clone(), aux = source.clone();
    while (no_iter--) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (aux.at<uchar>(i, j) == 0) {
                    for (int k = 0; k < 8; k++) {
                        int ni = i + n8_di[k];
                        int nj = j + n8_dj[k];
                        if (IsInside(rows, cols, ni, nj)) {
                            dst.at<uchar>(ni, nj) = 0;
                        }
                    }
                }
            }
        }
        aux = dst.clone();
    }
    return dst;

}
Mat color_labels(labels labels_str){

    int rows, cols, no_labels;
    Mat labelsM, result;
    Vec3b* colors;

    rows = labels_str.labelsM.rows;
    cols = labels_str.labelsM.cols;
    labelsM= labels_str.labelsM;
    no_labels = labels_str.no_labels;
    colors = new Vec3b[no_labels];

    default_random_engine gen;
    uniform_int_distribution<int> d(0,255);

    for (int i = 0; i < no_labels; i++) {
        uchar r = d(gen);
        uchar g = d(gen);
        uchar b = d(gen);
        colors[i] = Vec3b(b, g, r);
    }
    result = Mat(rows, cols, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int label = labelsM.at<int>(i, j);
            if (label > 0) {
                result.at<Vec3b>(i, j) = colors[label];
            }
        }
    }
    return result;
}
labels BFS_labeling(Mat source) {
    Mat labelsM;
    int rows, cols, no_labels;
    no_labels = 0;
    rows = source.rows, cols = source.cols;
    labelsM = Mat(rows, cols, CV_32SC1, Scalar(0));

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if (source.at<uchar>(i,j) == 0 && labelsM.at<int>(i,j) == 0) {
                no_labels++;
                std::queue<Point> Q;
                labelsM.at<int>(i,j) = no_labels;
                Q.push(Point(j,i));

                while (!Q.empty()) {
                    Point q = Q.front();
                    Q.pop();

                    for (int k = 0; k < 4; k++) {
                        int x = q.y + n4_di[k];
                        int y = q.x + n4_dj[k];

                        if (x >= 0 && x < rows && y >= 0 && y < cols) {
                            if (source.at<uchar>(x,y) == 0 && labelsM.at<int>(x,y) == 0) {
                                labelsM.at<int>(x,y) = no_labels;
                                Q.push(Point(y,x));
                            }
                        }
                    }
                }
            }
        }
    }
    return {labelsM, no_labels};
}

vector<Rect> detectEyes(Mat source) {

    Mat gray = bgr_2_grayscale(source);
    imshow("gray", gray);
    int* histogram = compute_histogram_naive(gray);
    gray = apply_histogram_equalization(gray, histogram);
    imshow("gray equalized", gray);
    Mat binary = threshold_inverse_binary(gray, 60);

    imshow("binary", binary);

    Mat invBinary;
    invBinary = negative_image(binary);

    imshow("invBinary", invBinary);

    invBinary = dilation(invBinary, 2);
    imshow("dilated", invBinary);

    labels labelsStruct = BFS_labeling(invBinary);
    Mat labelsMat = labelsStruct.labelsM;
    int no_labels = labelsStruct.no_labels;

    Mat colorLabels = color_labels(labelsStruct);
    imshow("labeled components", colorLabels);

    vector<Rect> eyes;

    for (int label = 1; label <= no_labels; label++) {
        int minX = invBinary.cols, minY = invBinary.rows;
        int maxX = 0, maxY = 0;
        int no_pixels = 0;
        Point center(0, 0);

        for (int y = 0; y < labelsMat.rows; y++) {
            for (int x = 0; x < labelsMat.cols; x++) {
                if (labelsMat.at<int>(y, x) == label) {
                    minX = min(minX, x);
                    minY = min(minY, y);
                    maxX = max(maxX, x);
                    maxY = max(maxY, y);
                    center.x += x;
                    center.y += y;
                    no_pixels++;
                }
            }
        }
        center.x /= no_pixels;
        center.y /= no_pixels;

        int width = maxX - minX;
        int height = maxY - minY;
        float aspectRatio = (float)width / height;

        if (aspectRatio >= 1 && aspectRatio <= 3 &&
            no_pixels > 500 && no_pixels < 1500) {
            Rect eyeRect(minX, minY, width, height);
            eyes.push_back(eyeRect);
        }
    }
    return eyes;
}


Mat displayEyeZones(Mat img, vector<Rect> eyes) {

    for (Rect eye : eyes) {
        rectangle(img, eye, Scalar(0, 0, 255), 2);
    }
    return img;
}

image_channels_bgr break_channels(Mat source){
    int rows, cols;
    Mat B, G, R;
    image_channels_bgr bgr_channels;

    rows = source.rows;
    cols = source.cols;
    B = Mat(rows, cols, CV_8UC1);
    G = Mat(rows, cols, CV_8UC1);
    R = Mat(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            B.at<uchar>(i,j) = source.at<Vec3b>(i, j)[0];
            G.at<uchar>(i,j) = source.at<Vec3b>(i, j)[1];
            R.at<uchar>(i,j) = source.at<Vec3b>(i, j)[2];
        }
    }
    bgr_channels.B = B;
    bgr_channels.G = G;
    bgr_channels.R = R;
    return bgr_channels;
}

image_channels_hsv bgr_2_hsv(image_channels_bgr bgr_channels){
    int rows, cols;
    Mat H, S, V;
    image_channels_hsv hsv_channels;

    rows = bgr_channels.B.rows;
    cols = bgr_channels.B.cols;
    Mat r,g,b;
    r = Mat(rows, cols, CV_32FC1);
    g = Mat(rows, cols, CV_32FC1);
    b = Mat(rows, cols, CV_32FC1);
    V = Mat(rows, cols, CV_32FC1);
    H = Mat(rows, cols, CV_32FC1);
    S = Mat(rows, cols, CV_32FC1);
    Mat M = Mat(rows, cols, CV_32FC1);
    Mat m = Mat(rows, cols, CV_32FC1);
    Mat C = Mat(rows, cols, CV_32FC1);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            float red = (float)bgr_channels.R.at<uchar>(i,j)/255;
            r.at<float>(i,j) = red;
            float green = (float)bgr_channels.G.at<uchar>(i,j)/255;
            g.at<float>(i,j) = green;
            float blue = (float)bgr_channels.B.at<uchar>(i,j)/255;
            b.at<float>(i,j) = blue;

            float maxx = M.at<float>(i,j) = max(max(red, green), blue);
            float minn = m.at<float>(i,j) = min(min(red, green), blue);
            C.at<float>(i,j) = maxx - minn;
            V.at<float>(i,j) = M.at<float>(i,j);

            if (V.at<float>(i,j) != 0)
                S.at<float>(i,j) = C.at<float>(i,j)/V.at<float>(i,j);
            else
                    S.at<float>(i,j) = 0;

            if (C.at<float>(i,j) != 0) {
                if (M.at<float>(i,j) == red)
                    H.at<float>(i,j) = 60*(green - blue)/C.at<float>(i,j);
                if (M.at<float>(i,j) == green)
                    H.at<float>(i,j) = 120 + 60*(blue - red)/C.at<float>(i,j);
                if (M.at<float>(i,j) == blue)
                    H.at<float>(i,j) = 240 + 60*(red - green)/C.at<float>(i,j);
            }
            else
                H.at<float>(i,j) = 0;

            if (H.at<float>(i,j) < 0)
                H.at<float>(i,j)+= 360;
        }
    }
    hsv_channels.H = H;
    hsv_channels.S = S;
    hsv_channels.V = V;

    return hsv_channels;
}


image_channels_bgr fixRedEyes(image_channels_bgr bgr_channels, image_channels_hsv hsv_channels, vector<Rect> eyes) {
    for (int i = 0; i < eyes.size(); ++i) {
        Rect eye = eyes[i];

        Mat eyeH = hsv_channels.H(eye);
        Mat eyeS = hsv_channels.S(eye);
        Mat eyeV = hsv_channels.V(eye);

        Mat maskH = ((eyeH >= 0) & (eyeH <= 15)) | ((eyeH >= 345) & (eyeH <= 360));
        Mat maskS = (eyeS > 0.7);
        Mat maskV = (eyeV > 0.5);
        Mat red_eye_mask = maskH & maskS & maskV;

        imshow("Red Eye Portion" + to_string(i), red_eye_mask);

        Mat eyeB = bgr_channels.B(eye);
        Mat eyeG = bgr_channels.G(eye);
        Mat eyeR = bgr_channels.R(eye);

        for (int x = 0; x < red_eye_mask.rows; x++) {
            for (int y = 0; y < red_eye_mask.cols; y++) {
                if (red_eye_mask.at<uchar>(x, y)) {
                    uchar colorOfPupil = (eyeB.at<uchar>(x, y) + eyeG.at<uchar>(x, y)) / 2;  //color of pupil
                    eyeB.at<uchar>(x, y) = eyeG.at<uchar>(x,y) = eyeR.at<uchar>(x,y) =  colorOfPupil;
                }
            }
        }
    }
    return bgr_channels;
}