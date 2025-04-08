// src/ImageProcessor.cpp

#include "ImageProcessor.hpp"
#include <opencv2/imgproc.hpp>

ImageProcessor::ImageProcessor(const std::string &imagePath) {
    originalImage = cv::imread(imagePath);
    if (originalImage.empty()) {
        throw std::runtime_error("Не удалось загрузить изображение: " + imagePath);
    }
}

void ImageProcessor::preprocess() {
    // Уменьшаем шумы и улучшаем цветовую сегментацию
    cv::Mat hsv;
    cv::cvtColor(originalImage, hsv, cv::COLOR_BGR2HSV);
    cv::GaussianBlur(hsv, hsv, cv::Size(5, 5), 0);
    cv::cvtColor(hsv, originalImage, cv::COLOR_HSV2BGR);
}

void ImageProcessor::createMasks() {
    cv::Mat gray, edges;
    cv::cvtColor(originalImage, gray, cv::COLOR_BGR2GRAY);
    
    // Детектор краёв
    cv::Canny(gray, edges, 50, 150);
    
    // Обнаружение линий (преобразование Хафа)
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10);
    
    // Создание маски зачеркивания
    maskOverwritten = cv::Mat::zeros(originalImage.size(), CV_8UC1);
    for (const auto& line : lines) {
        cv::line(maskOverwritten, cv::Point(line[0], line[1]), 
                 cv::Point(line[2], line[3]), cv::Scalar(255), 2);
    }
    
    // Маска текста: инверсия зачеркивания + морфология
    cv::bitwise_not(maskOverwritten, maskWritten);
    cv::morphologyEx(maskWritten, maskWritten, cv::MORPH_CLOSE, 
                    cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5)));
}

void ImageProcessor::removeBackground() {
    // в HSV и минус небелый фон
    cv::Mat hsv;
    cv::cvtColor(originalImage, hsv, cv::COLOR_BGR2HSV);
    
    // для белого цвета
    cv::inRange(hsv, cv::Scalar(0, 0, 200), cv::Scalar(255, 30, 255), backgroundMask);
    
    // фон = 1, объекты = 0
    cv::bitwise_not(backgroundMask, backgroundMask);
    
    // для очистки артефактов
    cv::morphologyEx(backgroundMask, backgroundMask, cv::MORPH_CLOSE, 
                    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));
}

void ImageProcessor::process() {
    preprocess();
    removeBackground();
    createMasks();
    originalImage.setTo(cv::Scalar(0, 0, 0), backgroundMask);
}

void ImageProcessor::saveMasks(const std::string &mask1Path, const std::string &mask2Path) {
    // Маска зачеркивания (бинарная)
    cv::imwrite(mask1Path, maskOverwritten);

    // Маска текста с исходными цветами
    cv::Mat textWithColor;
    originalImage.copyTo(textWithColor, maskWritten);
    cv::imwrite(mask2Path, textWithColor);
}