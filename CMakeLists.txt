cmake_minimum_required(VERSION 3.10)
project(ImageProcessing)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(ImageProcessing main.cpp src/ImageProcessor.cpp)

target_link_libraries(ImageProcessing ${OpenCV_LIBS})