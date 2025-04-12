#ifndef IMAGEPROCESSOR_HPP
#define IMAGEPROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <string>

class ImageProcessor
{
public:
    struct Config
    {
        int minOverwrittenArea; // Минимальная площадь зачеркивания
        int blurSize;           // Размер ядра для размытия

        // Конструктор по умолчанию
        Config() : minOverwrittenArea(500), blurSize(5) {}
    };

    // Конструктор с путем к изображению и конфигурацией
    ImageProcessor(const std::string &imagePath, const Config &config = Config());

    // Основной метод обработки
    void process();

    // Сохранение бинарных масок
    void saveMasks(const std::string &mask1Path, const std::string &mask2Path);

    // Оценка качества масок с помощью IoU
    double calculateIoU(const cv::Mat &mask1, const cv::Mat &mask2);

private:
    cv::Mat originalImage;   // Исходное изображение
    cv::Mat hsvImage;        // Изображение в HSV-пространстве
    cv::Mat maskOverwritten; // Бинарная маска зачеркивания
    cv::Mat maskWritten;     // Бинарная маска текста
    cv::Mat backgroundMask;  // Маска фона
    cv::Mat labels;          // Метки кластеризации (для K-Means)
    Config config_;          // Конфигурационные параметры

    // Предобработка изображения
    void preprocess();

    // Создание масок текста и зачеркивания
    void createMasks();

    // Удаление фона
    void removeBackground();

    // Вспомогательный метод для поиска самого светлого кластера
    int findBrightestCluster();
};

#endif // IMAGEPROCESSOR_HPP