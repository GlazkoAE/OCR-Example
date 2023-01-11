## Пример OCR с использованием CTC Loss

### Описание
Датасет - комбинация EMNIST (digits) и EMNIST (letters)\
Функция потерь - CTC Loss\
Метрика качества - accuracy

Для тестирования реализован алгоритм beam search при помощи ctc_decoder из torchaudio

### Результаты
Длина последовательности символов в одном экземпляре: 1-5\
Число эпох для обучения: 70\
Размер выборки обучающей на эпоху: 1 024 000\
Размер валидационной выборки на эпоху: 10 000
Batch size: 1024

Accuracy: 0.89

<details><summary>Примеры работы</summary>

![plot_0](output/plot_0.png)
![plot_1](output/plot_1.png)
![plot_2](output/plot_2.png)
![plot_3](output/plot_3.png)
![plot_4](output/plot_4.png)
![plot_5](output/plot_5.png)
![plot_6](output/plot_6.png)
![plot_7](output/plot_7.png)
![plot_8](output/plot_8.png)
![plot_9](output/plot_9.png)

</details>

### Запуск проекта
1. Выполнить сборку образа
```
docker build -t ocr .
```
2. Внутри контейнера запустить обучение
```
python train.py
```

### Исходный проект
* https://github.com/dredwardhyde/crnn-ctc-loss-pytorch