#include <opencv2/opencv.hpp>
#include <iostream>

int main(){
    cv::Mat img=cv::imread("OIP-C.webp",cv::IMREAD_COLOR);

cv::imshow("c++ opencv",img);
cv::waitKey(0);

cv::imwrite("new_image_1.jpg",img);
}
