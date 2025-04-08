// ImageProcessor.hpp
#ifndef IMAGEPROCESSOR_HPP
#define IMAGEPROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <string>

class ImageProcessor {
public:
    ImageProcessor(const std::string& imagePath);
    void process();
    void saveMasks(const std::string& mask1Path, const std::string& mask2Path);

private:
    cv::Mat originalImage;
    cv::Mat maskOverwritten;
    cv::Mat maskWritten;
    cv::Mat backgroundMask;

    void preprocess();
    void createMasks();
    void removeBackground();
};

#endif // IMAGEPROCESSOR_HPP