# misis2025s-course


## Для сборки проекта: 

cmake CMakeLists.txt -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake -B ./cmake-build-debug
/cmake-build-debug> make all

В моем случае:

cmake CMakeLists.txt -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=/Users/polinagorokhova/github/vcpkg/scripts/buildsystems/vcpkg.cmake -B ./cmake-build-debug
/cmake-build-debug> make all

## Запуск: 
(пример)

./ImageProcessing ../imgs/imgs_in/2.jpeg ../imgs/imgs_out/2_1.jpeg ../imgs/imgs_out/2_2.jpeg

./ImageProcessing ../imgs/imgs_in/1.jpeg ../imgs/imgs_out/1_1.jpeg ../imgs/imgs_out/1_2.jpeg
