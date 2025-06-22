# ПОДРОБНОЕ ОПИСАНИЕ ПРОЦЕССА РЕШЕНИЯ

## 1. Загрузка и предварительная обработка изображения

### 1.1 Загрузка изображения
```cpp
originalImage = cv::imread(imagePath);
```
- Форматы: PNG, JPEG
- Цветовое пространство: BGR (стандарт OpenCV)
- Размер: автоматически определяется из файла

### 1.2 Преобразование в HSV
```cpp
cv::cvtColor(originalImage, hsvImage, cv::COLOR_BGR2HSV);
```
- H (Hue): 0-180 (в OpenCV)
- S (Saturation): 0-255
- V (Value): 0-255

![HSV Conversion](imgs/hsv_conversion.png)
*Рисунок 1: Преобразование изображения в HSV*

### 1.3 Удаление шума
```cpp
// Гауссовское размытие
cv::GaussianBlur(hsvImage, hsvImage, cv::Size(3, 3), 0);

// Билатеральная фильтрация
cv::bilateralFilter(tempBGR, filteredBGR, 5, 50, 50);
```
Параметры:
- Гауссовское размытие:
  - Размер ядра: 3x3
  - Сигма: 0 (автоматический расчет)
- Билатеральная фильтрация:
  - Размер ядра: 5
  - Пространственная сигма: 50
  - Цветовая сигма: 50

![Noise Removal](imgs/noise_removal.png)
*Рисунок 2: Результат удаления шума*

## 2. Кластеризация цветов

### 2.1 Удаление фона
```cpp
cv::inRange(hsvImage, cv::Scalar(0, 0, 100), cv::Scalar(180, 30, 255), backgroundMask);
```
Параметры:
- Нижний порог: (0, 0, 100)
- Верхний порог: (180, 30, 255)

![Background Removal](imgs/background_removal.png)
*Рисунок 3: Результат удаления фона*

### 2.2 Подготовка данных для кластеризации
```cpp
struct PixelData {
    float h, s, v;  // HSV значения
    float x, y;     // Нормализованные координаты
};
```
- Нормализация координат: деление на размеры изображения
- Веса признаков:
  - HSV: 1.0
  - Координаты: 20.0

### 2.3 K-means кластеризация
```cpp
cv::kmeans(samples, clusterCount, labels,
           cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1),
           3, cv::KMEANS_PP_CENTERS, centers);
```
Параметры:
- Количество кластеров: 3
- Максимум итераций: 30
- Точность: 0.1
- Метод инициализации: k-means++

![Clustering Results](imgs/clustering.png)
*Рисунок 4: Результаты кластеризации*

## 3. Создание масок

### 3.1 Начальные маски
```cpp
maskWritten = cv::Mat::zeros(hsvImage.size(), CV_8U);
maskOverwritten = cv::Mat::zeros(hsvImage.size(), CV_8U);
```
- Тип: CV_8U (8-битное беззнаковое целое)
- Инициализация: нулевая матрица

### 3.2 Расширение масок
```cpp
cv::Mat expanded = expandMask(mask, hsv, refColor, maxDist, iterations);
```
Параметры:
- Максимальное расстояние: 30.0
- Количество итераций: 3
- Веса компонентов:
  - Оттенок: 2.0
  - Насыщенность: 2.0
  - Яркость: 1.0

![Mask Expansion](imgs/mask_expansion.png)
*Рисунок 5: Процесс расширения масок*

### 3.3 Уточнение границ
```cpp
cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
cv::morphologyEx(mask, refined, cv::MORPH_CLOSE, kernel);
```
Параметры:
- Размер ядра: 3x3
- Тип операции: CLOSE (закрытие)

![Boundary Refinement](imgs/boundary_refinement.png)
*Рисунок 6: Уточнение границ масок*

## 4. Сохранение результатов

### 4.1 Формат сохранения
```cpp
cv::imwrite(mask1Path, maskWritten);
cv::imwrite(mask2Path, maskOverwritten);
```
- Формат: PNG
- Тип: бинарные маски (0 и 255)

![Final Results](imgs/final_results.png)
*Рисунок 7: Итоговые маски*

## 5. Оптимизация производительности

### 5.1 Векторизация операций
- Использование матричных операций OpenCV
- Минимизация циклов
- Эффективное использование памяти

### 5.2 Параллельная обработка
- Автоматическое использование многопоточности OpenCV
- Оптимизация для многоядерных процессоров

## 6. Обработка ошибок

### 6.1 Проверка входных данных
```cpp
if (originalImage.empty()) {
    throw std::runtime_error("Failed to load image: " + imagePath);
}
```

### 6.2 Валидация параметров
- Проверка диапазонов значений
- Корректность путей к файлам
- Достаточность памяти

## 7. Метрики качества

### 7.1 Точность разделения
- Процент правильно классифицированных пикселей
- Метрика IoU (Intersection over Union)

### 7.2 Производительность
- Время обработки: ~100-500 мс на изображение
- Использование памяти: ~2-3x размер исходного изображения 