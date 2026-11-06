# Диффузионные методы переноса стиля — этап 1

## Quick Start

1. Сохраните файлы проекта в [репозитории](https://github.com/Batyaoftheyear/diffusion-style). Из локальной рабочей папки код автоматически не публикуется.
2. [Откройте colab_stage1.ipynb в Colab](https://colab.research.google.com/github/Batyaoftheyear/diffusion-style/blob/main/colab_stage1.ipynb), выберите **T4 GPU** и выполняйте ячейки сверху вниз.
3. Ноутбук подключит Drive, создаст отдельное Python-окружение, подготовит данные, запустит smoke test на 2 фото, затем baseline на 24 фото, метрики и PDF.
4. Просмотрите страницы отчёта и генерации. Для ЛМС нужны окончательный PDF и ссылка на GitHub.

**Текущий статус:** код первого этапа подготовлен; реальные GPU-генерации и DINO-оценка ещё не выполнены. Baseline не считается воспроизведённым. Локальный черновик PDF явно помечен и не готов к сдаче. Результаты доступных проверок и границы подтверждённого — в [docs/validation.md](docs/validation.md).

## Задача и архитектура

Сравнение стилевой LoRA с переносом через IP-Adapter планируется на втором этапе. Сейчас реализован только **Stable Diffusion 1.5 + ControlNet Canny + IP-Adapter**. ControlNet получает контуры исходного фото, IP-Adapter — одну картину, промпт описывает содержание фотографии. Начальное состояние — сохранённый шум; img2img не используется.

Есть один компактный ноутбук запуска; вся ML-логика расположена в `src/`, независимые команды — в `scripts/`. Обучение LoRA, перебор рангов и динамические коэффициенты описаны только в документах. Их реализации и дополнительных ноутбуков в проекте нет.

| Файл | Назначение |
|---|---|
| `configs/baseline.yaml` | Все настройки данных, моделей, генерации и метрик |
| `src/config.py`, `src/io.py`, `src/bootstrap.py` | Валидация, расположение артефактов, fingerprints, атомарная запись |
| `src/data.py`, `src/preprocessing.py` | Отбор, загрузка, RGB/crop/Canny, captions, manifests и контроль |
| `src/pipeline.py`, `src/inference.py` | Загрузка моделей, реальные latents, GPU-инференс и resume |
| `src/metrics.py`, `src/visualization.py` | Edge matching, DINOv2, CSV/JSON и сравнительные таблицы |
| `src/report.py`, `scripts/build_report.py` | PDF из Markdown с настоящими данными и проверкой страниц |
| `colab_stage1.ipynb` | Единственный Colab-ноутбук всего первого этапа |
| `tests/` | CPU-тесты и отдельно включаемый настоящий GPU smoke test |
| `docs/theory.md`, `docs/methodology.md` | Теория, точные определения и протокол |
| `docs/stage1_report.md`, `docs/stage2_plan.md` | Редактируемый отчёт и план будущей работы |

## Данные и модели

Стиль: 75 изображений импрессионизма из [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart), из них 70 предназначены для будущего обучения, 5 — отдельные референсы. Первый референс фиксирован для baseline. Контроль — проверка изображений, точных pixel-дубликатов и близких dHash-копий до разбиения. Captions картин формируются по жанровой метке; перед будущим обучением их необходимо просмотреть.

Тест: 24 фотографии COCO validation 2017. [Небольшое зеркало аннотаций](https://huggingface.co/datasets/phiyodr/coco2017) содержит IDs, настоящие captions, license_id и URL; скачиваются только выбранные фотографии. Шесть тематических групп задаются по captions, по четыре фото в группе. Полный COCO и многогигабайтные WikiArt shards не загружаются.

Преобразование: EXIF → RGB → центральный crop/resize с сохранением пропорций → 512 × 512. Для фото дополнительно grayscale → Canny (100, 200). Центральная обрезка может убрать важные объекты: сначала просмотрите preview.

| Компонент | Репозиторий |
|---|---|
| SD 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| ControlNet Canny | `lllyasviel/control_v11p_sd15_canny` |
| IP-Adapter | `h94/IP-Adapter`, `models/ip-adapter_sd15.safetensors` |
| CLIP vision | `h94/IP-Adapter`, `models/image_encoder` |
| DINOv2 | `facebook/dinov2-base`, последний CLS-токен |

Полные SHA revisions закреплены в YAML. Выбранные модели, лицензии, алгоритм отбора и ограничения подробно описаны в [методологии](docs/methodology.md). WikiArt не объявляется свободно лицензированным: карточка разрешает некоммерческие исследования, права на работы индивидуальны. У COCO сохраняются оригинальные license_id и ссылки. SD/ControlNet имеют условия OpenRAIL, IP-Adapter и выбранная конверсия DINOv2 — Apache-2.0 по карточкам. Изображения, веса и кэши в Git не добавляются.

## Окружение

Цель: Linux/Google Colab, Python **3.11–3.13**, T4 16 GB, FP16. Зависимости верхнего уровня закреплены в `requirements.txt`; PyTorch 2.8.0 и torchvision 0.23.0 используются из официального индекса CUDA 12.6. Полная GPU-совместимость подтверждается только smoke test в конкретном runtime. Не устанавливайте этот CUDA-файл зависимостей в основной Windows Python.

Ноутбук создаёт `/content/diffusion-style-env` через официальный автономный virtualenv 21.15.5 с закреплённым SHA-256; системный ensurepip не требуется. Если Python-файл остался после неудачной попытки, проверяется наличие pip и восстанавливается частичное окружение. Вывод pip и stderr команд отображаются в ячейке. Предустановленное окружение kernel не меняется. Скрипты выполняются отдельными процессами из venv, поэтому генерационные модели освобождают память до запуска DINO. CPU offload включён по умолчанию. Подготовьте место в Drive для кэша весов размером в несколько гигабайт; локально модели не скачиваются. Платных API нет, открытым моделям и датасетам не требуется токен в ноутбуке.

В чистом Linux/Colab-окружении установка отдельно от ноутбука:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

## Порядок запуска

Выполняйте из корня checkout. В Colab это уже делает ноутбук. Переменная `DIFFUSION_STYLE_HOME` задаёт корень данных и результатов; без неё используется корень проекта. Для кэша Hugging Face ноутбук отдельно задаёт `HF_HOME` в Drive.

```bash
python -m unittest discover -s tests -v
python scripts/prepare_data.py --config configs/baseline.yaml
python scripts/prepare_data.py --config configs/baseline.yaml --check-only
python scripts/run_baseline.py --config configs/baseline.yaml --limit 2
python scripts/run_baseline.py --config configs/baseline.yaml
python scripts/evaluate.py --config configs/baseline.yaml
python scripts/build_report.py --config configs/baseline.yaml --render
```

Каждый скрипт читает сохранённые данные самостоятельно. Не требуется переносить Python-переменные между процессами или ноутбуками. `--limit 2` — настоящий запуск SD + ControlNet + IP-Adapter, не мок. Успешные изображения smoke test используются основным baseline. Отдельный opt-in тест того же пути: `RUN_GPU_SMOKE=1 python -m unittest discover -s tests -p test_gpu_smoke.py -v` (Linux).

## Если prepare_data получает HTTP 502/503 от Hugging Face

Это ошибка удалённого Dataset Viewer при чтении каталога WikiArt. Обновите checkout через раздел 3 и повторите подготовку данных. Загрузчик делает до шести попыток: обычные паузы 2, 4, 8, 16 и 30 секунд; при Retry-After ожидание может увеличиваться до 60 секунд. В ячейке видны код ответа и следующий повтор.

Уже полученные страницы каталога сохраняются в Drive и повторно используются при той же закреплённой revision; для скачивания изображений просроченные временные ссылки обновляются отдельно. Успешные загрузки не удаляются, состав выборки и seed не меняются. Если все попытки исчерпаны, Viewer всё ещё недоступен: подготовку нужно повторить позднее. Переустановка пакетов, токен, удаление Drive и изменение GPU не исправляют ответ 503 сервера.

## Если PDF-тест не находит шрифт с кириллицей

Обновите checkout через раздел 3 и повторите ячейку CPU-тестов. Генератор использует DejaVu Sans и Bold из уже установленного Matplotlib, поэтому не зависит от наличия системного пакета шрифтов в Colab. Устанавливать дополнительные пакеты или задавать REPORT_FONT_PATH вручную не требуется. Переменная остаётся доступна для явного выбора своего TTF.

## Если Colab падает на создании окружения

Обновите notebook и checkout через раздел 3, затем повторите раздел 4. Вызов системного `python -m venv` заменён на автономный [virtualenv](https://virtualenv.pypa.io/en/latest/how-to/install.html#zipapp). Он устанавливает встроенный pip без системного ensurepip и исправляет уже созданное окружение без pip. Bootstrap-архив скачивается с официального GitHub PyPA с проверкой закреплённого SHA-256; системные Python-пакеты не изменяются. Удалять данные из Drive или перезапускать runtime для этого не нужно.

## Если Colab падает на git fetch

`CalledProcessError` с кодом 128 — только оболочка ошибки. Обновлённая ячейка выводит сообщение самого Git и отдельно распознаёт пустой remote. Если GitHub-репозиторий ещё не содержит коммитов, ветка main не существует: подготовленные файлы нужно сначала опубликовать из локальной папки проекта. Смена main на master эту ситуацию не исправит.

После первой публикации повторите ячейку загрузки репозитория. Удалять каталог в Colab или сбрасывать Google Drive для этого не требуется. При другой причине (авторизация, сеть, отсутствующий ref) ячейка покажет настоящий ответ Git.

## Настройки и метрики

Исходная точка: 512 × 512, 30 шагов, DDIM eta=0, CFG 7.5, ControlNet 1.0, IP-Adapter 0.6, seed 42, Canny 100/200, FP16. Значения не считаются оптимальными. Используется один общий seed на все фото и отдельный CPU-generator; реальный FP32 noise 1 × 4 × 64 × 64 сохраняется в NPY перед приведением к FP16.

- Композиция: Edge precision, recall и F1 с фиксированным Chebyshev-допуском 2 пикселя; два направления matching имеют разные знаменатели.
- Содержание: cosine similarity нормированных CLS-признаков DINOv2 при одном image processor.
- Стиль: визуальный анализ палитры, мазков и фактуры по таблицам Original / Canny / Style Reference / Generated.
- Ресурсы: время пайплайна после CUDA synchronization, peak allocated/reserved в МиБ; загрузка моделей отдельно.

`metrics.csv` содержит строки для 24 фото. `summary.json` — mean, sample SD (ddof=1) и n. При одном seed SD характеризует фотографии, а не устойчивость к шуму. На втором этапе запланированы 3 seed; сейчас дополнительного перебора нет.

## Где сохраняются результаты и как продолжить

При запуске в ноутбуке корень — `MyDrive/diffusion_style_transfer/`:

```text
data/<data_id>/
  originals/ prepared/ canny/ captions/ items/
  art_manifest.jsonl coco_manifest.jsonl manifest.json
  preview.png
results/<run_id>/
  run.json parameters.json
  images/ latents/ metadata/
  results.csv results.jsonl
  figures/comparisons.png figures/report_examples.png
  evaluation/<evaluation_id>/
  metrics.csv summary.json
results/current_run.json
reports/stage1_report.pdf
reports/stage1_report.json
reports/preview/page_*.png
.cache/huggingface/
```

`data_id` учитывает настройки и версии preprocessing. `run_id` учитывает конфигурацию, hashes входов, генерационный код и окружение/GPU. `evaluation_id` дополнительно учитывает код и параметры метрик. При изменении настроек создаётся новый прогон; результаты другой конфигурации не переиспользуются. Переиспользование проверяет hashes PNG и latents. Повреждённые или недописанные результаты повторяются. Зафиксированные удалённые исходники должны соответствовать прежнему SHA.

После прерывания повторите ячейки сверху вниз. Установка вenv потребуется в новом runtime, но подготовленные данные и проверенные генерации уже находятся в Drive. Не запускайте несколько runtimes с записью в один каталог. Если меняете алгоритм подготовки, увеличьте DATA_VERSION в src/config.py.

## PDF и готовность к сдаче

Текст находится в `docs/stage1_report.md`. Сборщик автоматически добавляет конфигурацию, реальные metrics и первые два заранее определённых примера. Без полного проверенного baseline и соответствующих метрик обычная сборка завершается ошибкой. Допустимый предварительный режим:

```bash
python scripts/build_report.py --draft --render
```

Он создаёт **`reports/stage1_report_draft.pdf`**, а не окончательный файл; каждая страница помечена «ЧЕРНОВИК». Число страниц проверяется (4–6), кириллица извлекается из PDF, страницы можно просмотреть как PNG. Самостоятельная визуальная проверка перед сдачей всё равно нужна. После успешного baseline и evaluate запустите обычную сборку для `reports/stage1_report.pdf`.

| Критерий | Подтверждение кодом/материалами | Что ещё подтвердить |
|---|---|---|
| Теория, 8 | theory.md, разделы отчёта | Просмотр и защита |
| Пайплайн, 11 | pipeline.py, inference.py, methodology.md | Совместный GPU-запуск |
| Baseline, 11 | Скрипт, конфиг, latents, resume, grids | 24 реальные генерации в Colab |
| План, 5 | stage2_plan.md и отчёт | Уточнение гиперпараметров во втором этапе |
| Метрики, 3 | CPU-проверки Edge/cosine, реализация DINO | Реальные DINO/CSV для 24 фото |
| План этапа 2, 2 | stage2_plan.md | Последующая исследовательская работа |

До реального Colab-прогона и просмотра окончательного PDF первый этап не завершён. Количество потенциальных баллов не является обещанием оценки.

## Источники

- [Latent Diffusion](https://arxiv.org/abs/2112.10752), [ControlNet](https://arxiv.org/abs/2302.05543), [IP-Adapter](https://arxiv.org/abs/2308.06721), [LoRA](https://arxiv.org/abs/2106.09685), [DINOv2](https://arxiv.org/abs/2304.07193).
- [Diffusers: IP-Adapter](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter), [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet), [исходники выбранной версии](https://github.com/huggingface/diffusers/tree/v0.35.2).
- [PyTorch 2.8: официальные варианты установки](https://pytorch.org/get-started/previous-versions/), [воспроизводимость](https://docs.pytorch.org/docs/stable/notes/randomness.html).
- Карточки данных, моделей и условия использования — в [docs/methodology.md](docs/methodology.md).
