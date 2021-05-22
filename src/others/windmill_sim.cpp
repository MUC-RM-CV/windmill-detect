#include <opencv2/opencv.hpp>
#include <iostream>

// Generate a frame with a given angle
cv::Mat generate_canvas(double angle) {
    cv::Mat canvas(500, 500, CV_8UC3);
    canvas = cv::Scalar(255,255,255);

    auto radius = 0.3 * canvas.cols;
    auto center = cv::Point2d(0.5 * canvas.cols, 0.5 * canvas.rows);
    
    auto delta_x = sin(angle) * radius;
    auto delta_y = cos(angle) * radius;
    cv::arrowedLine(canvas, center, center + cv::Point2d(delta_x, delta_y), cv::Scalar(0,255,0), 5);

    for (int i = 1; i <= 4; i++) {
        auto delta_x = sin(angle + CV_PI * 0.4 * i) * radius;
        auto delta_y = cos(angle + CV_PI * 0.4 * i) * radius;
        cv::line(canvas, center, center + cv::Point2d(delta_x, delta_y), cv::Scalar(0,255,0), 5);
    }

    return canvas;
}

int main() {
    double s_since_begin = 0;
    double angle = 0;
    double radius = 150;
    double fps = 40;
    
    cv::VideoWriter out;
    out.open("out.mp4", cv::VideoWriter::fourcc('H','2','6','4'), fps, cv::Size(500, 500));

    int cnt = 1000; // number of total frames to generate
    double passed_time_each_frame = 1.0 / fps;
    while (cnt > 0) {
        auto spd = 0.785 * sin(1.884 * s_since_begin) + 1.305;
        angle += passed_time_each_frame * spd;

        out.write(generate_canvas(angle)); 
        // cv::imshow("canvas", canvas);

        s_since_begin += passed_time_each_frame;
        --cnt;
    }
}
