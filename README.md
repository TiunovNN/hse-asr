# Automatic Speech Recognition (ASR) with PyTorch

Студент: Тиунов Н. Н.

Отчет: https://api.wandb.ai/links/tiunovnn-t/1m1cjq1f

# Предварительная настройка

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements
```

# Запуск обучения

```shell
python train.py -cn=conformer3
```

# Запуск инференса


```shell
python download_language.py
python inference.py -cn=inference_conformer3
```

# Полученные результаты

| Датасет | Метрика | Среднее значение |
| -------- | --------- | ------------ |
| clean | CER | 0.119 |
| clean | WER | 0.278 |
| other | CER | 0.294 |
| other | WER | 0.565 |
