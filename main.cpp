// main.cpp

#include <iostream>
#include "ImageProcessor.hpp"
#include <filesystem>

void saveIntermediateResults(const cv::Mat &image, const std::string &name)
{
    std::string path = "debug_images/" + name + ".png";
    cv::imwrite(path, image);
    std::cout << "Saved intermediate result: " << path << std::endl;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Использование: " << argv[0] << " <путь_к_изображению> <путь_к_маске_зачеркивания> <путь_к_маске_записи>" << std::endl;
        return 1;
    }

    try
    {
        // Создаем директорию для отладочных изображений
        std::__fs::filesystem::create_directories("debug_images");

        ImageProcessor processor(argv[1]);

        // Сохраняем оригинальное изображение
        saveIntermediateResults(processor.getOriginalImage(), "1_original");

        // Запускаем обработку
        processor.process();

        // Сохраняем промежуточные результаты
        saveIntermediateResults(processor.getHSVImage(), "2_hsv_converted");
        saveIntermediateResults(processor.getPreprocessedImage(), "3_preprocessed");
        saveIntermediateResults(processor.getBackgroundMask(), "4_background_mask");
        saveIntermediateResults(processor.getTextMask(), "5_text_mask");
        saveIntermediateResults(processor.getStrikethroughMask(), "6_strikethrough_mask");

        // Сохраняем финальные маски
        processor.saveMasks(argv[2], argv[3]);
        std::cout << "Обработка завершена. Маски сохранены." << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Ошибка: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}