#include "WindmillDetect.h"

using std::vector;
using cv::Point;

void WindmillDetect::drawTetragon(cv::Mat &image, cv::Point2f *vertices, const cv::Scalar &color)  {
    using cv::Scalar;
    int thickness = (int)ceil(5e-3 * image.cols);
    for (int j = 0; j < 4; j++) {
        cv::line(image, vertices[j], vertices[(j + 1) % 4], color, thickness);
    }
    int radius = (int)ceil(1e-2 * image.cols);
    //cv::circle(image, vertices[0], radius, Scalar(  0,   0, 255), -1); // red
    //cv::circle(image, vertices[1], radius, Scalar(  0, 255, 255), -1); // yellow
    //cv::circle(image, vertices[2], radius, Scalar(255,   0, 255), -1); // purple / violet
    //cv::circle(image, vertices[3], radius, Scalar(255,   0,   0), -1); // blue
}

std::vector<cv::Point2f> WindmillDetect::process(const cv::Mat &frame) {

    std::vector<cv::Mat> channels;

    split(frame, channels);

    gray = channels[2] - channels[0]; // Extract Red

    threshold(gray, binary, 60, 255, cv::THRESH_BINARY);
    dilate(binary, binary, element);

    vector<vector<Point>> contours;
    //vector<Vec4i> hierarchy;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    if (this->draw_result) {
        frame.copyTo(show);
    }

    auto frame_area = frame.cols * frame.rows;

    //cv::Point2f vertices[4];

    vector<cv::RotatedRect> rrects;
    vector<double> rrect_sizes;
    for (auto & c : contours) {
        auto r = minAreaRect(c);
        rrect_sizes.push_back(r.size.width * r.size.height);
        rrects.push_back(r);
    }

    std::vector<cv::Point2f> aimArea;

    bool found_fan = false;
    for (int i = 0; i < contours.size(); ++i) {

        auto rrect = rrects[i];

        auto area = rrect_sizes[i];
        if (area < 1e-3 * frame_area || area > 8e-2 * frame_area) continue;

        double hu_moments[7];
        HuMoments(moments(contours[i]), hu_moments);
        // for (auto i : hu_moments) { std::cout << i << ' '; } std::cout << '\n';

        double d1 = myDistance(to_hit_hu_moments, hu_moments);
        double d2 = myDistance(hit_hu_moments, hu_moments);

        // std::cout << "d1: " << d1 << "\t";
        // std::cout << "d2: " << d2 << "\t";

        const double max_diff = 0.15; // Max diff allowed for myDistance

        if (d1 < max_diff && d2 > max_diff) {
            found_fan = true;
            // std::cout << "-> to_hit  " << "Contour area: " << area << "\n";
        } else {
            found_fan = false;
            // std::cout << "-> others  " << "Contour area: " << area << "\n";
        }

        if (this->draw_result) {
            cv::Point2f srcPt[4];
            rrect.points(srcPt);
            drawTetragon(show, srcPt, cv::Scalar(238, 165, 65));
        }

        if (found_fan) {
            for (int j = 0; j < contours.size(); ++j) {
                if (rrect_sizes[j] > 0.3 * area || rrect_sizes[j] < 0.08 * area) continue;
                std::vector<cv::Point2f> vertices;
                if (rotatedRectangleIntersection(rrects[j], rrects[i], vertices) == cv::INTERSECT_FULL) {
                    cv::Point2f pts[4];
                    rrects[j].points(pts);
                    
                    if (this->draw_result) {
                        cv::Point2f pos = (((pts[0] + pts[2]) / 2) + ((pts[1] + pts[3]) / 2)) / 2;
                        cv::circle(show, pos, 5, cv::Scalar(0,0,255));
                        drawTetragon(show, pts, cv::Scalar(255,255,255));
                    }

                    aimArea = {pts[0], pts[1], pts[2], pts[3]};
                    break;
                }
            }
        }


        /* // Show Fan; can be used for preparing image for SVM training
        int height = rrect.size.height;
        int width = rrect.size.width;

        cv::Mat straighten_rect;
        Point2f dstPt[4];

        if (rrect.size.width > rrect.size.height) {
            straighten_rect = cv::Mat(height, width, CV_8UC1); // rows, cols
            dstPt[0] = Point2f(0, height - 1);          // -> bottom left of dst rect
            dstPt[1] = Point2f(0, 0);                   // -> top left of dst rect
            dstPt[2] = Point2f(width - 1, 0);           // -> top right of dst rect
            dstPt[3] = Point2f(width - 1, height - 1);  //  -> bottom right of dst rect
        } else {
            // straighten_rect = cv::Mat(width, height, CV_8UC1);
            dstPt[0] = Point2f(0, 0);                   // -> top left of dst rect
            dstPt[1] = Point2f(height - 1, 0);          // -> top right of dst rect
            dstPt[2] = Point2f(height - 1, width - 1);  // -> bottom right of dst rect
            dstPt[3] = Point2f(0, width - 1);           // -> bottom left of dst rect
        }

        // std::cout << straighten_rect.rows << " * " << straighten_rect.cols << " ";
        // std::cout << rrect.size.width << " * " << rrect.size.height << "\n";

        cv::Mat M = cv::getPerspectiveTransform(srcPt, dstPt);
        cv::warpPerspective(binary, straighten_rect, M, straighten_rect.size());
        cv::imshow("fan", straighten_rect); */
    }

    return aimArea;
}
