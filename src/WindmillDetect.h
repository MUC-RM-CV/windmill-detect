#ifndef MUC_CLEMENTINE_WINDMILLDETECT_H
#define MUC_CLEMENTINE_WINDMILLDETECT_H

#define DEBUG

#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

class WindmillDetect {
private:
    cv::Mat element;

    // 距离会导致二值图的变化，根据距离需要改变是否开运算，最后需要修改图像矩
    double to_hit_hu_moments[7] = {
            0.34038, 0.05055, 0.03539, 0.01653, 0.000166, 0.00350, 0
    };

    double hit_hu_moments[7] = {
            0.21294, 0.01848, 4.43766E-06, 5.53527E-06, 2.80028E-11, 7.44906E-07, -1.72424E-12
    };

    static auto myDistance(double A[7], double B[7]) {
        double sum = 0;
        for (int i = 0; i < 7; ++i) { sum += abs(A[i] - B[i]); }
        return sum;
    }

public:
    cv::Mat gray;
    cv::Mat binary;
    cv::Mat show;

    void setElement(const cv::Mat& e) { element = e; }

    std::vector<cv::Point2f> process(const cv::Mat& frame);

    static void drawTetragon(cv::Mat& image, cv::Point2f *vertices, const cv::Scalar & color);
};

#endif //MUC_CLEMENTINE_WINDMILLDETECT_H
