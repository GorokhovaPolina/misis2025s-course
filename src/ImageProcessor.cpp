#include "ImageProcessor.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdexcept>
#include <iostream>
#include <vector>

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
    cv::cvtColor(originalImage, hsvImage, cv::COLOR_BGR2HSV);
    cv::GaussianBlur(hsvImage, hsvImage, cv::Size(3, 3), 0);
    cv::Mat debugBGR;
    cv::cvtColor(hsvImage, debugBGR, cv::COLOR_HSV2BGR);
    cv::imwrite("debug_preprocessed.png", debugBGR);
}

void ImageProcessor::removeBackground()
{
    cv::Mat rgbData;
    originalImage.reshape(1, originalImage.rows * originalImage.cols).convertTo(rgbData, CV_32F);
    cv::kmeans(rgbData, 3, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 1.0),
               5, cv::KMEANS_PP_CENTERS);

    labels = labels.reshape(1, originalImage.rows);

    std::vector<int> clusterCounts(3, 0);
    for (int i = 0; i < labels.rows; i++)
    {
        for (int j = 0; j < labels.cols; j++)
        {
            int label = labels.at<int>(i, j);
            clusterCounts[label]++;
        }
    }

    int backgroundLabel = std::distance(clusterCounts.begin(),
                                        std::max_element(clusterCounts.begin(), clusterCounts.end()));

    backgroundMask = cv::Mat::zeros(originalImage.size(), CV_8UC1);
    for (int i = 0; i < originalImage.rows; i++)
    {
        for (int j = 0; j < originalImage.cols; j++)
        {
            if (labels.at<int>(i, j) != backgroundLabel)
            {
                backgroundMask.at<uchar>(i, j) = 255;
            }
        }
    }

    cv::morphologyEx(backgroundMask, backgroundMask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    cv::imwrite("debug_background_mask.png", backgroundMask);

    if (cv::countNonZero(backgroundMask) == 0)
    {
        std::cerr << "Предупреждение: backgroundMask пуста!" << std::endl;
    }
}

void ImageProcessor::createMasks()
{
    // Увеличиваем количество кластеров до 4
    int numClusters = 4;
    cv::Mat rgbData;
    originalImage.reshape(1, originalImage.rows * originalImage.cols).convertTo(rgbData, CV_32F);
    cv::kmeans(rgbData, numClusters, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 1.0),
               5, cv::KMEANS_PP_CENTERS);
    labels = labels.reshape(1, originalImage.rows);

    // Создаем маски для каждого кластера
    std::vector<cv::Mat> clusterMasks(numClusters);
    for (int k = 0; k < numClusters; k++)
    {
        clusterMasks[k] = cv::Mat::zeros(originalImage.size(), CV_8UC1);
        for (int i = 0; i < originalImage.rows; i++)
        {
            for (int j = 0; j < originalImage.cols; j++)
            {
                if (labels.at<int>(i, j) == k)
                {
                    clusterMasks[k].at<uchar>(i, j) = 255;
                }
            }
        }
        cv::imwrite("debug_cluster_" + std::to_string(k) + ".png", clusterMasks[k]);
    }

    // Определяем кластер фона как самый большой
    int backgroundLabel = -1;
    int maxCount = 0;
    for (int k = 0; k < numClusters; k++)
    {
        int count = cv::countNonZero(clusterMasks[k]);
        if (count > maxCount)
        {
            maxCount = count;
            backgroundLabel = k;
        }
    }

    // Оставшиеся кластеры — объекты (текст и зачеркивания)
    std::vector<int> objectClusters;
    for (int k = 0; k < numClusters; k++)
    {
        if (k != backgroundLabel)
        {
            objectClusters.push_back(k);
        }
    }

    // Инициализируем маски
    maskWritten = cv::Mat::zeros(originalImage.size(), CV_8UC1);     // Маска текста
    maskOverwritten = cv::Mat::zeros(originalImage.size(), CV_8UC1); // Маска зачеркиваний

    // Анализируем каждый кластер объектов
    for (int idx : objectClusters)
    {
        cv::Mat cluster = clusterMasks[idx];
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(cluster, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto &contour : contours)
        {
            cv::Rect bbox = cv::boundingRect(contour);
            double aspectRatio = static_cast<double>(bbox.width) / bbox.height;
            double area = cv::contourArea(contour);

            // Проверяем форму: зачеркивания обычно вытянутые
            bool isOverwritten = (aspectRatio > 3.0 || 1.0 / aspectRatio > 3.0) && area > 50;

            // Анализируем текстуру с помощью фильтра Габора
            cv::Mat roi = originalImage(bbox);
            cv::Mat grayRoi;
            cv::cvtColor(roi, grayRoi, cv::COLOR_BGR2GRAY);
            cv::Mat gaborResponse;
            cv::Mat gaborKernel = cv::getGaborKernel(cv::Size(5, 5), 1.0, 0.0, 1.0, 1.0);
            cv::filter2D(grayRoi, gaborResponse, CV_32F, gaborKernel);
            double meanResponse = cv::mean(gaborResponse)[0];

            // Если отклик Габора низкий и форма вытянутая — это зачеркивание
            if (isOverwritten && meanResponse < 50)
            {
                cv::drawContours(maskOverwritten, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), -1);
            }
            else
            {
                cv::drawContours(maskWritten, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), -1);
            }
        }
    }

    // Очищаем маску текста от зачеркиваний
    cv::Mat notOverwritten;
    cv::bitwise_not(maskOverwritten, notOverwritten);
    cv::bitwise_and(maskWritten, notOverwritten, maskWritten);

    // Сохраняем отладочные изображения
    cv::imwrite("debug_overwritten_mask.png", maskOverwritten);
    cv::imwrite("debug_text_mask.png", maskWritten);
}

void ImageProcessor::process()
{
    preprocess();
    removeBackground();
    createMasks();
    cv::Mat notBackgroundMask;
    cv::bitwise_not(backgroundMask, notBackgroundMask);
    originalImage.setTo(cv::Scalar(0, 0, 0), notBackgroundMask);
    cv::imwrite("debug_final_image.png", originalImage);
}

void ImageProcessor::saveMasks(const std::string &mask1Path, const std::string &mask2Path)
{
    cv::imwrite(mask1Path, maskOverwritten);
    cv::imwrite(mask2Path, maskWritten);
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
    return 0;
}