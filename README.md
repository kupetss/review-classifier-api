# Review Classifier API

![tests](https://github.com/kupetss/review-classifier-api/actions/workflows/tests.yml/badge.svg)

REST API которое определяет тональность отзыва на фильм: положительный или отрицательный

## Стек
python, scikit-learn, FastAPI, Pydantic, pytest, Docker, GitHub Actions

## Результат
Модель: TF-IDF + логистическая регрессия.
- TF-IDF превращает текст в числа: каждое слово получает вес
  - TF — как часто слово встречается в отзыве
  - IDF — насколько оно редкое по всем отзывам
- Логистическая регрессия по этим весам решает, хороший отзыв или плохой

Обучена на датасете [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb): 25 000 отзывов для обучения, 25 000 для теста

| Метрика | Значение |
|---|---|
| Accuracy | 0.895 |
| F1 (negative / positive) | 0.89 / 0.90 |

## Запуск
Через Docker: 
```bash
docker build -t review-classifier .
docker run -p 8000:8000 review-classifier
```

Локально:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Документация API: http://127.0.0.1:8000/docs

## Обучение модели заново
```bash
python -m ml.load_data
python -m ml.train
```

## Пример запроса
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "An amazing film, I loved it!"}'
```

Ответ:
```json
{"sentiment": "positive", "probability": 0.97}
```
probability - насколько модель уверена в своём ответе, от 0 до 1

## Ограничения модели
- Модель не понимает отрицания и контекст: «good, no bad» распознаётся как отрицательный отзыв с уверенностью 99%, потому что слово `bad` имеет больший вес, чем `good`.
- Смайлики игнорируются: TF-IDF учитывает только слова.
- Модель обучена на английских отзывах о фильмах и на других текстах будет работать хуже.