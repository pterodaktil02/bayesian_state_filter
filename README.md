# Bayesian State Filter for Home Assistant

Версия: **0.4.0**

[English README](README.en.md)

`bayesian_state_filter` - пользовательская интеграция Home Assistant для объединения нескольких асинхронных числовых датчиков в одну робастную оценку скрытого скалярного процесса.

В отличие от обычного усреднения, фильтр одновременно оценивает:

- уровень процесса;
- локальную скорость изменения;
- ускорение;
- рывок;
- относительный bias каждого источника;
- эффективную sigma каждого источника;
- доверие к каждому новому измерению;
- доверие к производным динамической модели.

Типичные применения: температура, давление, влажность, радиационный фон и другие непрерывные или квазинепрерывные величины, измеряемые несколькими источниками в совместимых единицах.

> Не путать со встроенной интеграцией Home Assistant `bayesian`: она оценивает вероятность события и формирует бинарный сенсор. Этот проект оценивает непрерывное числовое состояние.

## Модель состояния

Состояние фильтра всегда имеет вид:

```text
[x, v, a, j]
```

где `x` - уровень, `v` - скорость, `a` - ускорение, `j` - jerk.

Модель использует интегрированный Wiener-процесс с шумом snap. Для среднего и ковариации применяется одна и та же confidence-gated матрица перехода. Это важно для численной устойчивости: скрытые производные не могут разогнать covariance через негейтированные перекрестные связи.

### Доверие к производным

Для каждой производной вычисляется posterior z-score:

```text
z_d = |d| / sigma_d
```

Из него получается вероятность того, что параметр отличен от нуля при нормальном posterior:

```text
c(z) = erf(|z| / sqrt(2))
```

Эффективные веса иерархические:

```text
w_v = c_v
w_a = c_v * c_a
w_j = c_v * c_a * c_j
```

Поэтому всегда выполняется:

```text
1 >= w_v >= w_a >= w_j >= 0
```

Смысл простой: если скорость не наблюдается уверенно, ускорение и jerk не должны сильнее влиять на траекторию.

### Робастное объединение источников

Каждое измерение обновляет общий state отдельно. Источники не сводятся заранее к среднему.

Для каждого источника оцениваются:

- относительный `bias`;
- эффективная `sigma`;
- типичный cadence;
- innovation;
- z-score innovation;
- Student-t robust weight;
- статистика выбросов.

Student-t updater допускает вес согласованного inlier-измерения немного выше 1 и плавно уменьшает вес выбросов.

### Обучение динамики

`q/timescale` идентифицируются по истории Recorder на естественной временной сетке процесса. Валидация идет на 1-2 естественных шага, а не на произвольный длинный горизонт.

RMSE используется как диагностический показатель качества идентификации, но не определяет confidence производных. Confidence берется только из posterior state/covariance.

### Быстрый рестарт и checkpoint

Состояние сохраняется в Home Assistant Store каждые **30 минут**.

Checkpoint содержит:

- `[x, v, a, j]`;
- covariance `P`;
- `q/timescale`;
- калибровки источников;
- characteristic-time diagnostics;
- Recorder watermarks;
- служебную версию schema.

После рестарта интеграция:

```text
load checkpoint
-> read Recorder tail after saved watermark
-> replay tail through the normal filter path
-> live mode
```

Полное обучение по нескольким суткам истории выполняется только если checkpoint отсутствует, поврежден или несовместим по schema. Online-калибровка источников не пересчитывается на каждом sample: тяжелый refit запускается адаптивно по состоянию drift-monitor.

## Online-калибровка источников

На каждом новом измерении выполняются обычное обновление фильтра и дешевый drift-monitor. Переоценка `bias/sigma` по историческому окну выполняется отдельно и адаптивно: редко для стабильных источников и чаще при признаках дрейфа.

Drift оценивается относительно собственного baseline источника, а не общего фиксированного порога. Состояние scheduler сохраняется в checkpoint.

Для общего bias поддерживаются три режима `bias_anchor`:

- `median` - робастный относительный нулевой gauge;
- `mean` - линейный sum-to-zero gauge;
- `passport` - абсолютная привязка по паспортной точности моделей; каждая модель дает один робастный голос независимо от числа одинаковых физических датчиков.

## Установка

Скопируйте каталог:

```text
custom_components/bayesian_state_filter
```

в:

```text
/config/custom_components/bayesian_state_filter
```

После замены Python-файлов выполните полный перезапуск Home Assistant.

YAML можно перечитать без полного перезапуска действием:

```text
bayesian_state_filter.reload
```

## Минимальная конфигурация

```yaml
sensor:
  - platform: bayesian_state_filter
    name: "Температура зала"
    ensemble:
      sources:
        - sensor.temperature_1
        - sensor.temperature_2
        - sensor.temperature_3
```

Все источники должны измерять одну физическую величину в совместимых единицах. Автоматического преобразования единиц нет.

## Рекомендуемая конфигурация

```yaml
sensor:
  - platform: bayesian_state_filter
    name: "Температура зала"

    ensemble:
      sources:
        - sensor.temperature_1
        - sensor.temperature_2
        - sensor.temperature_3
        - sensor.temperature_4
      min_sources: 2

    bayes:
      noise_model: auto
      history_days: 7
      save_every_s: 1800
      student_nu: 4
      characteristic_refit_s: 21600
      diagnostics: compact
```

Основные параметры:

| Параметр | По умолчанию | Назначение |
|---|---:|---|
| `history_days` | `7` | История Recorder для startup-калибровки и идентификации динамики |
| `save_every_s` | `1800` | Интервал checkpoint |
| `student_nu` | `4` | Число степеней свободы Student-t updater |
| `characteristic_refit_s` | `21600` | Период переоценки характерного времени уровня |
| `warmup_refit_s` | `21600` | Минимальный период повторного обучения динамики, пока она еще не идентифицирована |
| `bias_anchor` | `median` | Gauge для общего bias: `median`, `mean` или `passport` |
| `noise_model` | `auto` | `auto`, `gaussian` или `poisson` |
| `diagnostics` | `compact` | `compact`, `full`, `debug`, `verbose` |

## Основные атрибуты

Итоговый sensor публикует, в частности:

```text
stddev
rate_per_hour
curvature_per_hour2
jerk_per_hour3

rate_weight
curvature_weight
jerk_weight

rate_z
curvature_z
jerk_z

gated_timescale_s
gated_local_rmse
gated_local_rmse_step1
gated_local_rmse_step2
```

`rate/curvature/jerk` публикуются в удобных единицах на час, хотя внутри модели производные хранятся в SI-времени на секунду.

Для каждого источника в `source_health` доступны:

```text
model
bias
sigma
median_dt_s
outlier_rate
last_z_score
robust_weight
```

Также сохраняется отдельная оценка `characteristic_time_s`. Она описывает медленную динамику уровня процесса и не обязана совпадать с `gated_timescale_s`, который относится к локальной state-space модели.

## Практические замечания

- `bias` относительный: без внешнего эталона фильтр не знает общей систематической ошибки всех датчиков.
- `sigma` - эффективная ошибка наблюдения общего скрытого состояния, а не паспортная точность датчика.
- Высокий `robust_weight` означает согласованное измерение, низкий - выброс или временное расхождение источника.
- Большие значения `curvature_per_hour2` или `jerk_per_hour3` сами по себе не означают аварию: они являются локальными производными, пересчитанными из секунд в часы. Смотреть их нужно вместе с `curvature_weight`, `jerk_weight` и локальным RMSE.
- Если динамика неразличима на фоне шума, корректное поведение - веса производных стремятся к нулю и фильтр фактически становится оценкой уровня.

## Структура

```text
custom_components/
  bayesian_state_filter/
    __init__.py
    const.py
    manifest.json
    sensor.py
    services.yaml
    strings.json
    translations/
    core/
      filter.py
      gated_training.py
      noise_models.py
      process_noise.py
      state_models.py
      training.py
      types.py
      updaters.py
      variogram.py
```

## Лицензия

Проект распространяется по GNU GPL-3.0-only.
