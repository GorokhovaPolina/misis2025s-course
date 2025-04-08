// main.cpp

#include <iostream>
#include "ImageProcessor.hpp"

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Использование: " << argv[0] << " <путь_к_изображению> <путь_к_маске_зачеркивания> <путь_к_маске_записи>" << std::endl;
        return 1;
    }

    try
    {
        ImageProcessor processor(argv[1]);
        processor.process();
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