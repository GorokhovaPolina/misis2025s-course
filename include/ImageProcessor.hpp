#ifndef IMAGEPROCESSOR_HPP
#define IMAGEPROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <string>

class ImageProcessor
{
public:
    ImageProcessor(const std::string &imagePath);
    void process();
    void saveMasks(const std::string &mask1Path, const std::string &mask2Path);

    // Геттеры для доступа к промежуточным результатам
    const cv::Mat &getOriginalImage() const { return originalImage; }
    const cv::Mat &getHSVImage() const { return hsvImage; }
    const cv::Mat &getPreprocessedImage() const { return preprocessedImage; }
    const cv::Mat &getBackgroundMask() const { return backgroundMask; }
    const cv::Mat &getTextMask() const { return maskWritten; }
    const cv::Mat &getStrikethroughMask() const { return maskOverwritten; }

private:
    void preprocess();
    void clusterColors(int k = 3);
    cv::Mat expandMask(const cv::Mat &mask, const cv::Mat &hsv, const cv::Vec3f &refColor, float maxDist, int iterations);

    cv::Mat originalImage;
    cv::Mat hsvImage;
    cv::Mat preprocessedImage; // Добавляем поле для хранения предобработанного изображения
    cv::Mat backgroundMask;
    cv::Mat maskWritten;     // Text mask
    cv::Mat maskOverwritten; // Strikethrough mask
};

#endif // IMAGEPROCESSOR_HPP