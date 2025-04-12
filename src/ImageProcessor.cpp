#include "ImageProcessor.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdexcept>
#include <iostream>

ImageProcessor::ImageProcessor(const std::string &imagePath, const Config &config)
    : config_(config)
{
    originalImage = cv::imread(imagePath);
    if (originalImage.empty())
    {
        throw std::runtime_error("Не удалось загрузить изображение: " + imagePath);
    }
}

void ImageProcessor::preprocess()
{
    // Перевод в HSV
    cv::cvtColor(originalImage, hsvImage, cv::COLOR_BGR2HSV);
    // Минимальное размытие для удаления мелкого шума
    cv::GaussianBlur(hsvImage, hsvImage, cv::Size(3, 3), 0);
    // Отладка
    cv::Mat debugBGR;
    cv::cvtColor(hsvImage, debugBGR, cv::COLOR_HSV2BGR);
    cv::imwrite("debug_preprocessed.png", debugBGR);
}

void ImageProcessor::removeBackground()
{
    // Сегментация фона в HSV (белый фон)
    cv::inRange(hsvImage, cv::Scalar(0, 0, 200), cv::Scalar(255, 30, 255), backgroundMask);

    // Инверсия: объекты = 255, фон = 0
    cv::bitwise_not(backgroundMask, backgroundMask);

    // Удаление мелкого шума
    cv::morphologyEx(backgroundMask, backgroundMask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    // Отладка
    cv::imwrite("debug_background_mask.png", backgroundMask);

    // Проверка
    if (cv::countNonZero(backgroundMask) == 0)
    {
        std::cerr << "Предупреждение: backgroundMask пуста!" << std::endl;
    }
}

void ImageProcessor::createMasks()
{
    // 1. Выделение текста (черного) через цветовую сегментацию
    cv::Mat textMask;
    // Черный цвет в HSV: низкая яркость
    cv::inRange(hsvImage, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 50), textMask);

    // Улучшение текста через морфологию
    cv::morphologyEx(textMask, textMask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    // Применяем маску фона
    cv::bitwise_and(textMask, backgroundMask, maskWritten);

    // Отладка
    cv::imwrite("debug_text_mask_initial.png", maskWritten);

    if (cv::countNonZero(maskWritten) == 0)
    {
        std::cerr << "Предупреждение: maskWritten пуста после цветовой сегментации!" << std::endl;
    }

    // 2. Выделение зачеркивания (красного) через цветовую сегментацию
    cv::Mat redMask;
    // Красный цвет в HSV: оттенок около 0 или 180
    cv::Mat redMask1, redMask2;
    cv::inRange(hsvImage, cv::Scalar(0, 50, 50), cv::Scalar(10, 255, 255), redMask1);    // Нижний красный
    cv::inRange(hsvImage, cv::Scalar(170, 50, 50), cv::Scalar(180, 255, 255), redMask2); // Верхний красный
    redMask = redMask1 | redMask2;

    // Улучшение зачеркивания
    cv::morphologyEx(redMask, redMask, cv::MORPH_DILATE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

    // Применяем маску фона
    cv::bitwise_and(redMask, backgroundMask, maskOverwritten);

    // Отладка
    cv::imwrite("debug_overwritten_mask.png", maskOverwritten);

    if (cv::countNonZero(maskOverwritten) == 0)
    {
        std::cerr << "Предупреждение: maskOverwritten пуста после цветовой сегментации!" << std::endl;
    }

    // 3. Очистка маски текста от зачеркивания
    cv::Mat notOverwritten;
    cv::bitwise_not(maskOverwritten, notOverwritten);
    cv::bitwise_and(maskWritten, notOverwritten, maskWritten);

    // Отладка
    cv::imwrite("debug_text_mask_final.png", maskWritten);
}

void ImageProcessor::process()
{
    preprocess();
    removeBackground();
    createMasks();
    // Применяем маску фона (фон становится черным)
    cv::Mat notBackgroundMask;
    cv::bitwise_not(backgroundMask, notBackgroundMask);
    originalImage.setTo(cv::Scalar(0, 0, 0), notBackgroundMask);
    // Отладка
    cv::imwrite("debug_final_image.png", originalImage);
}

void ImageProcessor::saveMasks(const std::string &mask1Path, const std::string &mask2Path)
{
    cv::imwrite(mask1Path, maskOverwritten); // Зачеркивание
    cv::imwrite(mask2Path, maskWritten);     // Текст
}

double ImageProcessor::calculateIoU(const cv::Mat &mask1, const cv::Mat &mask2)
{
    cv::Mat intersection, union_;
    cv::bitwise_and(mask1, mask2, intersection);
    cv::bitwise_or(mask1, mask2, union_);
    double intersectionCount = cv::countNonZero(intersection);
    double unionCount = cv::countNonZero(union_);
    return unionCount > 0 ? intersectionCount / unionCount : 0.0;
}

int ImageProcessor::findBrightestCluster()
{
    return 0; // Не используется
}