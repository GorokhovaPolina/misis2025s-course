#include "ImageProcessor.hpp"
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <opencv2/core/types.hpp>

// Custom structure to hold HSV and position data
struct PixelData
{
    float h, s, v; // HSV values
    float x, y;    // Normalized position

    PixelData(float h_, float s_, float v_, float x_, float y_)
        : h(h_), s(s_), v(v_), x(x_), y(y_) {}
};

ImageProcessor::ImageProcessor(const std::string &imagePath)
{
    originalImage = cv::imread(imagePath);
    if (originalImage.empty())
    {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }
}

void ImageProcessor::preprocess()
{
    // Convert to HSV
    cv::cvtColor(originalImage, hsvImage, cv::COLOR_BGR2HSV);
    if (hsvImage.empty())
    {
        throw std::runtime_error("Failed to convert image to HSV");
    }

    // Step 1: Noise reduction
    cv::GaussianBlur(hsvImage, hsvImage, cv::Size(3, 3), 0);
    if (hsvImage.empty())
    {
        throw std::runtime_error("Gaussian blur failed");
    }

    // Apply bilateral filter with adjusted parameters
    cv::Mat tempBGR;
    cv::cvtColor(hsvImage, tempBGR, cv::COLOR_HSV2BGR);
    cv::Mat filteredBGR;
    cv::bilateralFilter(tempBGR, filteredBGR, 5, 50, 50);

    // Store the preprocessed image
    preprocessedImage = filteredBGR.clone();

    // Convert back to HSV for further processing
    cv::cvtColor(filteredBGR, hsvImage, cv::COLOR_BGR2HSV);
}

cv::Mat ImageProcessor::expandMask(const cv::Mat &mask, const cv::Mat &hsv, const cv::Vec3f &refColor, float maxDist, int iterations)
{
    cv::Mat expanded = mask.clone();
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    for (int iter = 0; iter < iterations; ++iter)
    {
        cv::Mat border;
        cv::morphologyEx(expanded, border, cv::MORPH_DILATE, kernel);
        cv::subtract(border, expanded, border);

        for (int i = 0; i < hsv.rows; ++i)
        {
            for (int j = 0; j < hsv.cols; ++j)
            {
                if (border.at<uchar>(i, j) > 0)
                { // Исправлено условие
                    cv::Vec3b pixel = hsv.at<cv::Vec3b>(i, j);
                    float hueDiff = std::min(std::abs(pixel[0] - refColor[0]), 180 - std::abs(pixel[0] - refColor[0]));
                    float satDiff = abs(pixel[1] - refColor[1]);
                    float valDiff = abs(pixel[2] - refColor[2]);

                    // Weighted distance (more emphasis on saturation)
                    float dist = hueDiff * 1.0 + satDiff * 3.0 + valDiff * 1.0;
                    if (dist < maxDist)
                    {
                        expanded.at<uchar>(i, j) = 255;
                    }
                }
            }
        }
    }
    return expanded;
}

void ImageProcessor::clusterColors(int k)
{
    // Step 1: Remove the background
    cv::Mat foregroundMask;
    cv::inRange(hsvImage, cv::Scalar(0, 0, 100), cv::Scalar(180, 30, 255), backgroundMask);
    cv::bitwise_not(backgroundMask, foregroundMask);

    // Enhanced noise removal with larger kernel
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(foregroundMask, foregroundMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(foregroundMask, foregroundMask, cv::MORPH_CLOSE, kernel);

    // Additional noise removal using connected components
    cv::Mat labelsCC, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(foregroundMask, labelsCC, stats, centroids, 8, CV_32S);
    for (int label = 1; label < numLabels; label++)
    {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < 20) // Increased minimum area threshold
        {
            foregroundMask.setTo(0, labelsCC == label);
        }
    }

    // Step 2: Extract foreground pixels with position information
    std::vector<PixelData> foregroundPixels;
    for (int i = 0; i < hsvImage.rows; i++)
    {
        for (int j = 0; j < hsvImage.cols; j++)
        {
            if (foregroundMask.at<uchar>(i, j) == 255)
            {
                cv::Vec3b pixel = hsvImage.at<cv::Vec3b>(i, j);
                foregroundPixels.emplace_back(
                    pixel[0], pixel[1], pixel[2],
                    static_cast<float>(j) / hsvImage.cols,
                    static_cast<float>(i) / hsvImage.rows);
            }
        }
    }

    if (foregroundPixels.empty())
    {
        throw std::runtime_error("No foreground pixels found after background removal");
    }

    // Step 3: Enhanced clustering with position information
    cv::Mat samples(foregroundPixels.size(), 5, CV_32F);
    for (size_t i = 0; i < foregroundPixels.size(); i++)
    {
        samples.at<float>(i, 0) = foregroundPixels[i].h;         // H
        samples.at<float>(i, 1) = foregroundPixels[i].s;         // S
        samples.at<float>(i, 2) = foregroundPixels[i].v;         // V
        samples.at<float>(i, 3) = foregroundPixels[i].x * 20.0f; // Reduced position weight
        samples.at<float>(i, 4) = foregroundPixels[i].y * 20.0f; // Reduced position weight
    }

    int clusterCount = std::min(k, 4);
    cv::Mat labels, centers;
    cv::kmeans(samples, clusterCount, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1),
               3, cv::KMEANS_PP_CENTERS, centers);

    // Find two largest clusters with improved selection
    std::vector<int> counts(clusterCount, 0);
    std::vector<float> avgSaturation(clusterCount, 0.0f);
    std::vector<float> avgY(clusterCount, 0.0f);
    std::vector<float> avgHue(clusterCount, 0.0f);

    for (int i = 0; i < labels.rows; i++)
    {
        int label = labels.at<int>(i);
        counts[label]++;
        avgSaturation[label] += foregroundPixels[i].s;
        avgY[label] += foregroundPixels[i].y;
        avgHue[label] += foregroundPixels[i].h;
    }

    // Calculate average saturation for each cluster
    for (int i = 0; i < clusterCount; i++)
    {
        if (counts[i] > 0)
        {
            avgSaturation[i] /= counts[i];
            avgY[i] /= counts[i];
            avgHue[i] /= counts[i];
        }
    }

    // Sort clusters by size and saturation
    std::vector<int> sortedIndices(clusterCount);
    for (int i = 0; i < clusterCount; i++)
        sortedIndices[i] = i;
    std::sort(sortedIndices.begin(), sortedIndices.end(),
              [&counts, &avgSaturation, &avgY, &avgHue](int a, int b)
              {
                  if (counts[a] != counts[b])
                      return counts[a] > counts[b];
                  if (std::abs(avgSaturation[a] - avgSaturation[b]) > 15.0f)
                      return avgSaturation[a] < avgSaturation[b];
                  if (std::abs(avgY[a] - avgY[b]) > 0.1f)
                      return avgY[a] < avgY[b];
                  return avgHue[a] < avgHue[b];
              });

    cv::Vec3f color1(centers.at<float>(sortedIndices[0], 0),
                     centers.at<float>(sortedIndices[0], 1),
                     centers.at<float>(sortedIndices[0], 2));
    cv::Vec3f color2(centers.at<float>(sortedIndices[1], 0),
                     centers.at<float>(sortedIndices[1], 1),
                     centers.at<float>(sortedIndices[1], 2));

    // Determine which color is strikethrough based on saturation and position
    if (color1[1] > color2[1])
    {
        std::swap(color1, color2);
    }

    // Step 4: Create initial masks with improved distance calculation
    maskWritten = cv::Mat::zeros(hsvImage.size(), CV_8U);
    maskOverwritten = cv::Mat::zeros(hsvImage.size(), CV_8U);

    for (int i = 0; i < hsvImage.rows; i++)
    {
        for (int j = 0; j < hsvImage.cols; j++)
        {
            if (foregroundMask.at<uchar>(i, j) == 255)
            {
                cv::Vec3b pixel = hsvImage.at<cv::Vec3b>(i, j);

                // Enhanced color distance calculation with adjusted weights
                float hueWeight = 2.0f; // Increased weight for hue
                float satWeight = 2.0f; // Decreased weight for saturation
                float valWeight = 1.0f; // Kept the same

                float dist1 = std::min(std::abs(pixel[0] - color1[0]), 180 - std::abs(pixel[0] - color1[0])) * hueWeight +
                              abs(pixel[1] - color1[1]) * satWeight +
                              abs(pixel[2] - color1[2]) * valWeight;

                float dist2 = std::min(std::abs(pixel[0] - color2[0]), 180 - std::abs(pixel[0] - color2[0])) * hueWeight +
                              abs(pixel[1] - color2[1]) * satWeight +
                              abs(pixel[2] - color2[2]) * valWeight;

                // Add position-based bias for strikethrough with reduced influence
                float yPos = static_cast<float>(i) / hsvImage.rows;
                float strikethroughBias = std::exp(-(yPos - 0.5f) * (yPos - 0.5f) * 8.0f); // Reduced bias strength
                dist1 *= (1.0f - strikethroughBias * 0.1f);                                // Reduced bias influence

                if (dist1 < dist2)
                {
                    maskOverwritten.at<uchar>(i, j) = 255;
                }
                else
                {
                    maskWritten.at<uchar>(i, j) = 255;
                }
            }
        }
    }

    // Step 5: Enhanced mask refinement
    // Expand masks with different parameters for text and strikethrough
    maskWritten = expandMask(maskWritten, hsvImage, color2, 6.0f, 2);         // Reduced expansion
    maskOverwritten = expandMask(maskOverwritten, hsvImage, color1, 8.0f, 2); // Reduced expansion

    // Special handling for strikethrough with smaller kernel
    cv::Mat strikethroughKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 2)); // Smaller kernel
    cv::morphologyEx(maskOverwritten, maskOverwritten, cv::MORPH_CLOSE, strikethroughKernel);

    // Clean up text mask
    cv::Mat textKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
    cv::morphologyEx(maskWritten, maskWritten, cv::MORPH_CLOSE, textKernel);
    cv::morphologyEx(maskWritten, maskWritten, cv::MORPH_OPEN, textKernel);

    // Remove small components from both masks
    cv::Mat smallKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
    cv::morphologyEx(maskWritten, maskWritten, cv::MORPH_OPEN, smallKernel);
    cv::morphologyEx(maskOverwritten, maskOverwritten, cv::MORPH_OPEN, smallKernel);

    // Additional refinement: remove strikethrough from text mask
    cv::Mat tempMask;
    cv::bitwise_and(maskWritten, maskOverwritten, tempMask);
    cv::bitwise_xor(maskWritten, tempMask, maskWritten);

    // Final cleanup: remove border artifacts
    cv::Mat borderKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(maskWritten, maskWritten, cv::MORPH_ERODE, borderKernel);
    cv::morphologyEx(maskOverwritten, maskOverwritten, cv::MORPH_ERODE, borderKernel);

    // Debug: Save masks
    cv::imwrite("debug_text_mask.png", maskWritten);
    cv::imwrite("debug_strikethrough_mask.png", maskOverwritten);
}

void ImageProcessor::process()
{
    preprocess();
    clusterColors(3); // Using 3 clusters for better color separation

    // Set background to black in original image
    cv::Mat notBackgroundMask;
    cv::bitwise_not(backgroundMask, notBackgroundMask);
    originalImage.setTo(cv::Scalar(0, 0, 0), notBackgroundMask);
    cv::imwrite("debug_final_image.png", originalImage);
}

void ImageProcessor::saveMasks(const std::string &mask1Path, const std::string &mask2Path)
{
    cv::imwrite(mask1Path, maskOverwritten); // Strikethrough
    cv::imwrite(mask2Path, maskWritten);     // Text
}