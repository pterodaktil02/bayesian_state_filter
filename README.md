# Bayesian State Filter for Home Assistant

Версия: **0.2.1.6**

[English README](README.en.md)

`bayesian_state_filter` — пользовательская интеграция Home Assistant для объединения нескольких числовых сенсоров в одну робастную оценку общего скрытого состояния.

Основная задача интеграции — не просто усреднить несколько датчиков, а учитывать:

- постоянные относительные смещения (`bias`) источников;
- различную фактическую ошибку наблюдения (`sigma`);
- разную частоту обновления датчиков;
- асинхронные измерения;
- выбросы и временно "уехавшие" источники;
- динамику самого измеряемого процесса;
- историю Recorder до перезапуска Home Assistant.

Типичный пример — несколько датчиков температуры в одной комнате, но компонент не привязан к температуре. Он может использоваться с любыми числовыми источниками, если они измеряют одну и ту же физическую величину в совместимых единицах.

> Важно: интеграция **не выполняет преобразование единиц**. Все источники одного экземпляра фильтра должны быть совместимы по физическому смыслу и масштабу.

---

## 1. Как устроен фильтр

Внутреннее состояние двухмерное:

```text
x[0] = уровень процесса
x[1] = локальная скорость изменения
```

Для каждого нового измерения выполняется отдельный Bayesian update. Источники не схлопываются заранее в одно среднее измерение.

Модель скорости затухающая:

```text
dx/dt = v
dv/dt = -v/tau + process_noise
```

Здесь `tau` внутренней модели скорости — это **память локального тренда**, а не физическая постоянная времени объекта.

Отдельно оценивается `characteristic_time_s` — характерное время изменения **уровня процесса** по робастной временной вариограмме. Эти две величины имеют разный смысл и не должны интерпретироваться как одно и то же.

### Робастность к выбросам

Update выполняется через Student-t модель. Для нормального измерения вес близок к 1. Для сильного выброса вес уменьшается плавно, а не бинарно.

При `student_nu = 4` вес приблизительно определяется как:

```text
w = min(1, (nu + 1) / (nu + z^2))
```

где `z` — нормированная инновация.

Например, измерение с `z = 6.6` получает вес порядка `0.1` и почти не может утянуть итоговое состояние за собой.

---

## 2. Установка

Скопируйте каталог:

```text
custom_components/bayesian_state_filter
```

в:

```text
/config/custom_components/bayesian_state_filter
```

После установки или обновления выполните **полный Restart Home Assistant**.

Конфигурация в версии 0.2.1.6 — YAML platform. Config Flow пока не используется.

---

## 3. Минимальная конфигурация

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

При такой конфигурации используются все значения по умолчанию.

Если `sensor:` подключается через `!include_dir_merge_list`, файл может начинаться сразу так:

```yaml
- platform: bayesian_state_filter
  name: "Температура зала"
  ensemble:
    sources:
      - sensor.temperature_1
      - sensor.temperature_2
      - sensor.temperature_3
```

---

## 4. Рекомендуемая конфигурация

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
      save_every_s: 60
      tau_points: 16
      characteristic_refit_s: 1800
      student_nu: 4
      diagnostics: compact
```

Для большинства задач этого достаточно.

---

# 5. Полный справочник конфигурации

## 5.1 Верхний уровень

| Параметр | Тип | По умолчанию | Допустимые значения | Назначение |
|---|---:|---:|---|---|
| `platform` | string | — | `bayesian_state_filter` | Имя sensor platform. Обязательный параметр Home Assistant. |
| `name` | string | `Bayesian Sensor` | Любая строка | Имя создаваемого sensor entity. Для нескольких экземпляров фильтра используйте разные имена. |
| `ensemble` | mapping | `{}` | См. ниже | Описание набора исходных сенсоров. |
| `bayes` | mapping | `{}` | См. ниже | Параметры обучения, робастного update, динамики и диагностики. |

---

## 5.2 `ensemble`

### `ensemble.sources`

```yaml
ensemble:
  sources:
    - sensor.temperature_1
    - sensor.temperature_2
    - sensor.temperature_3
```

**Тип:** список entity ID.

**Практически обязательный параметр.** Код допускает пустой список, но фильтр тогда не получит измерений и не сможет работать.

Все источники должны:

- отдавать числовое состояние;
- измерять одну физическую величину;
- использовать совместимые единицы;
- иметь примерно один смысл скрытого состояния.

`unknown`, `unavailable`, `None`, нечисловые и не finite значения игнорируются.

Метаданные итогового sensor (`unit_of_measurement`, `device_class`, `state_class`, `icon`) наследуются от первого пригодного live-источника.

### `ensemble.min_sources`

```yaml
ensemble:
  min_sources: 2
```

**Тип:** integer  
**По умолчанию:** `1`  
**Минимум:** `1`

Минимальное количество **свежих** источников, необходимое для live update.

Источник считается свежим примерно в течение:

```text
max(3 * median_dt_source,
    0.20 * characteristic_time_or_velocity_tau,
    300 s)
```

Это позволяет редким датчикам участвовать в ансамбле без требования синхронного обновления всех источников.

Если свежих источников меньше `min_sources`, новое измерение сохраняется в warmup history, но основной Bayesian update не выполняется.

### Рекомендация по `min_sources`

- `1` — максимальная доступность; фильтр продолжает работать при одиночном живом датчике.
- `2` — хороший вариант для многосенсорной температуры: одиночный источник не ведет фильтр в полном одиночестве.
- `3+` — имеет смысл только если источников много и потеря нескольких датчиков допустима.

---

# 5.3 Блок `bayes`

Все параметры ниже находятся **внутри `bayes:`**.

```yaml
bayes:
  history_days: 7
  student_nu: 4
```

Параметры `save_every_s`, `student_nu` и остальные не должны находиться на верхнем уровне sensor.

---

## `bayes.noise_model`

```yaml
bayes:
  noise_model: auto
```

**Тип:** string  
**По умолчанию:** `auto`

Допустимые значения:

| Значение | Поведение в 0.2.1.6 |
|---|---|
| `auto` | Сейчас консервативно выбирает `gaussian`. Интерфейс уже подготовлен для будущего автоматического определения Пуассоновского шума. |
| `gaussian` | Постоянная для данного источника дисперсия наблюдения `sigma_i^2`. |
| `poisson` | Дисперсия пропорциональна текущему уровню сигнала. Эффективный коэффициент автоматически оценивается из обученной `sigma` и типичного уровня источника. |

Неизвестное значение вызывает warning и заменяется на `auto`.

### Gaussian

Для источника `i`:

```text
R_i = sigma_i^2
```

где `sigma_i` обучается автоматически по истории и затем медленно обновляется online.

### Poisson

Публичный тип шума остается просто `poisson`. Масштаб преобразования счетов в физическую величину не создает отдельного типа вроде `scaled_poisson`.

В текущей реализации:

```text
R_i(y) = k_i * |y|
k_i ~= sigma_i^2 / typical_abs_level_i
```

Пользователь не настраивает `k` вручную.

> В 0.2.1.6 `auto` еще не определяет Пуассон автоматически. Для счетного сигнала Poisson надо выбрать явно.

---

## `bayes.history_days`

```yaml
bayes:
  history_days: 7
```

**Тип:** float  
**По умолчанию:** `7.0` суток  
**Минимум:** `0.1` суток

Сколько истории каждого источника запрашивать из Home Assistant Recorder при старте.

История используется для:

- оценки относительного `bias` источников;
- pairwise-калибровки `sigma`;
- pretraining фильтра;
- оценки характерного времени уровня;
- обучения внутренней модели локальной скорости.

Большая история полезна для устойчивой оценки смещений и динамики, но текущая `sigma` не вычисляется по всей истории целиком. Для нее используется более свежее окно, чтобы давно прошедшие режимы не портили качество текущего источника.

### Практические значения

- `1` — быстрый старт, мало данных для медленной динамики.
- `3` — приемлемо для многих бытовых процессов.
- `7` — рекомендуемое общее значение.
- `14+` — может быть полезно для очень медленных процессов, но увеличивает startup processing.

Внутренний fused grid ограничивается примерно 20 000 точками; при необходимости шаг автоматически увеличивается.

---

## `bayes.save_every_s`

```yaml
bayes:
  save_every_s: 60
```

**Тип:** float  
**По умолчанию:** `60` секунд  
**Минимум:** `5` секунд

Минимальный интервал между сохранениями состояния фильтра в Home Assistant Store.

Сохраняются:

- posterior state и covariance;
- обученные source calibrations;
- внутренняя модель динамики;
- оценка characteristic time;
- шаг fused history grid.

Слишком маленькое значение увеличивает количество записей на диск без заметной пользы.

---

## `bayes.tau_points`

```yaml
bayes:
  tau_points: 16
```

**Тип:** integer  
**По умолчанию:** `16`  
**Минимум:** `8`

Количество точек логарифмической сетки для **внутренней predictive velocity model**.

Это не количество точек для итогового `characteristic_time_s` один к одному. Вариограмма использует не менее 32 точек и обычно `3 * tau_points`.

Большее значение:

- дает более плотную сетку `velocity_tau`;
- немного увеличивает startup CPU;
- обычно не нужно для бытовой телеметрии.

Рекомендуется оставить `16`.

---

## `bayes.tau_min_s`

```yaml
bayes:
  tau_min_s: 60
```

**Тип:** положительный float или отсутствует  
**По умолчанию:** auto

Нижняя граница `tau` **внутренней модели памяти локальной скорости**.

Не относится напрямую к `characteristic_time_s`.

Если параметр не задан, нижняя граница рассчитывается автоматически примерно как:

```text
max(4 * fused_grid_step,
    4 * fastest_source_median_dt,
    10 s)
```

Нулевое, отрицательное, не finite или `null` значение трактуется как отсутствие ручной границы.

---

## `bayes.tau_max_s`

```yaml
bayes:
  tau_max_s: 86400
```

**Тип:** положительный float или отсутствует  
**По умолчанию:** auto

Верхняя граница `tau` predictive velocity model.

Автоматическая верхняя граница примерно:

```text
max(8 * auto_tau_min,
    history_span / 2)
```

Если ручной `tau_max_s` меньше `tau_min_s`, внутренне он будет поднят чуть выше нижней границы.

---

## `bayes.characteristic_tau_min_s`

```yaml
bayes:
  characteristic_tau_min_s: 300
```

**Тип:** положительный float или отсутствует  
**По умолчанию:** auto

Нижняя граница поиска **характерного времени уровня процесса** по временной вариограмме.

Поддерживается старый алиас:

```yaml
characteristic_time_min_s: 300
```

Если заданы оба, приоритет у `characteristic_tau_min_s`.

Без ручной границы поиск начинается ниже минимального доступного lag, примерно от:

```text
min_lag / 4
```

Границу стоит задавать только если есть физически обоснованный диапазон. Слишком жесткая граница может сделать результат `boundary_limited`.

---

## `bayes.characteristic_tau_max_s`

```yaml
bayes:
  characteristic_tau_max_s: 172800
```

**Тип:** положительный float или отсутствует  
**По умолчанию:** auto

Верхняя граница поиска characteristic time уровня процесса.

Поддерживается алиас:

```yaml
characteristic_time_max_s: 172800
```

Без ручной границы поиск автоматически продолжается выше максимального наблюдаемого lag, чтобы отличить реально измеренное `tau` от ситуации "процесс медленнее доступной истории".

Если posterior упирается в верхнюю границу, `characteristic_time_status` становится `longer_than_history` и `characteristic_time_identifiable` будет `false`.

---

## `bayes.characteristic_refit_s`

```yaml
bayes:
  characteristic_refit_s: 1800
```

**Тип:** float  
**По умолчанию:** `1800` секунд  
**Минимум:** `300` секунд

Минимальный интервал между повторными online-оценками `characteristic_time_s`.

Оценка выполняется в background task по накопленной fused level history.

Рекомендуется:

- `1800` — нормальный общий вариант;
- `3600` — для очень медленных процессов и слабого CPU;
- ниже `300` задать нельзя: значение будет поднято до 300 секунд.

---

## `bayes.forget_time_s`

```yaml
bayes:
  forget_time_s: 259200
```

**Тип:** положительный float или отсутствует  
**По умолчанию:** auto

Время забывания для **банка predictive dynamics**, который обучает `velocity_tau` и `process_noise_q`.

Это не окно pairwise-калибровки source `sigma` и не период Recorder history.

Если не задано:

```text
forget_time_s = max(3 days, 8 * tau_max)
```

Если задано вручную, фактическое значение не может быть меньше шага fused grid.

Для обычного использования рекомендуется оставить auto.

---

## `bayes.student_nu`

```yaml
bayes:
  student_nu: 4
```

**Тип:** float  
**По умолчанию:** `4.0`  
**Минимум:** `1.01`

Число степеней свободы Student-t robust updater.

Интерпретация:

- меньше `nu` -> более тяжелые хвосты -> сильнее подавляются большие выбросы;
- больше `nu` -> поведение ближе к Gaussian update;
- `4` — хороший компромисс для бытовых сенсоров.

Robust update в 0.2.1.6 включен **всегда**. Параметра `robust_enable` нет и он не нужен.

---

## `bayes.diagnostics`

```yaml
bayes:
  diagnostics: compact
```

**Тип:** string  
**По умолчанию:** `compact`

Допустимые рабочие значения:

| Значение | Результат |
|---|---|
| `compact` | Рекомендуемый публичный набор атрибутов. |
| `full` | Добавляет внутренние исследовательские параметры. |
| `debug` | Алиас `full`. |
| `verbose` | Алиас `full`. |

Любая другая строка фактически ведет себя как `compact`.

`diagnostics` влияет только на количество показываемых атрибутов и **не меняет математику фильтра**.

---

# 6. Полный пример конфигурации

```yaml
sensor:
  - platform: bayesian_state_filter
    name: "Температура зала"

    ensemble:
      sources:
        - sensor.temp_1
        - sensor.temp_2
        - sensor.temp_3
        - sensor.temp_4
      min_sources: 2

    bayes:
      noise_model: auto
      history_days: 7
      save_every_s: 60
      tau_points: 16

      # Границы predictive velocity memory, обычно не нужны:
      # tau_min_s: 60
      # tau_max_s: 86400

      # Границы characteristic time уровня, обычно не нужны:
      # characteristic_tau_min_s: 300
      # characteristic_tau_max_s: 172800

      characteristic_refit_s: 1800

      # Обычно лучше оставить auto:
      # forget_time_s: 259200

      student_nu: 4
      diagnostics: compact
```

---

# 7. Что происходит при старте

При полном старте Home Assistant компонент:

1. Запрашивает Recorder history каждого source за `history_days`.
2. Очищает нечисловые и некорректные точки.
3. Оценивает типичный интервал каждого источника (`median_dt_s`).
4. Строит временной fused grid.
5. Оценивает относительный `bias` каждого источника по робастному пространственному reference.
6. Получает предварительный characteristic time уровня.
7. Выбирает свежее calibration window для `sigma`.
8. Для каждой пары источников совмещает близкие по времени измерения.
9. По pairwise residuals оценивает:

   ```text
   Var(i - j) ~= sigma_i^2 + sigma_j^2
   ```

10. Решает неотрицательную систему для отдельных `sigma_i`.
11. Перестраивает fused history уже с обученными source variances.
12. Оценивает итоговый `characteristic_time_s`.
13. Обучает predictive dynamics bank (`velocity_tau`, `process_noise_q`).
14. Проигрывает fused history через Bayesian filter, поэтому после нормального старта warmup почти не требуется.
15. Сохраняет обученное состояние в Home Assistant Store.

Если Recorder history недоступна, компонент пытается восстановить сохраненное состояние. Если и сохраненного состояния нет, он стартует от текущих live-измерений и обучается по мере накопления данных.

---

# 8. Как оцениваются `bias` и `sigma`

## `bias`

`bias` — относительное смещение источника относительно общего ансамбля.

Коррекция выполняется так:

```text
corrected = raw - bias
```

Например:

```text
raw  = 23.24
bias = +0.41
corrected = 22.83
```

`bias` не является абсолютной метрологической калибровкой. Если все датчики одновременно смещены на +0.5 единицы, ансамбль сам по себе не узнает абсолютную истину без внешнего эталона.

## `sigma`

`source_health.<source>.sigma` — это **эффективная ошибка наблюдения общего скрытого состояния**, а не паспортная точность микросхемы сенсора.

В нее могут входить:

- собственный шум датчика;
- квантование;
- локальный микроклимат в точке установки;
- тепловая инерция корпуса;
- небольшая ошибка временного совмещения;
- остаточная несогласованность источника с общей латентной величиной.

Поэтому датчик с паспортной точностью `+-0.1 C` может иметь `sigma = 0.25 C`, если его место установки систематически живет немного своей жизнью относительно остальной комнаты.

### Pairwise calibration

Для двух близких по времени измерений общая динамика процесса в первом приближении сокращается:

```text
Var(sensor_i - sensor_j) ~= sigma_i^2 + sigma_j^2
```

Для N источников получается переопределенная система из парных сравнений. Для пяти датчиков максимум доступно десять пар, а каждый отдельный источник может участвовать в четырех парах.

Медленный источник используется как anchor пары. Это предотвращает повторное использование одного медленного измерения десятки раз только потому, что соседний сенсор обновляется часто.

Для защиты от численного схлопывания `sigma -> 0` используется нижняя граница, связанная со статистическим разрешением pairwise variance estimates.

---

# 9. Characteristic time

Для уровня процесса используется робастная временная полувариограмма:

```text
gamma(h) = nugget + process_variance * (1 - exp(-h / tau))
```

где:

- `nugget` — высокочастотный/измерительный вклад;
- `process_variance` — амплитуда движения скрытого процесса;
- `tau` — characteristic time уровня.

Это **наблюдаемая характеристика данных**, а не обязательно чистая физическая постоянная времени объекта.

Например, в замкнутом контуре климатического регулирования `characteristic_time_s` отражает совокупность:

- помещения;
- контроллера;
- кондиционера;
- наружного воздействия;
- режима эксплуатации.

## `characteristic_time_status`

Возможные значения:

| Статус | Смысл |
|---|---|
| `identified` | Характерное время различимо по имеющимся данным. |
| `insufficient_signal` | Движение уровня слишком мало относительно noise/nugget; `tau` не имеет физически устойчивой оценки. |
| `below_resolution` | Характерное время ниже временного разрешения доступных данных или упирается в нижнюю границу. |
| `longer_than_history` | Процесс медленнее доступного диапазона lag/history; видна скорее нижняя оценка времени, чем полная идентификация. |
| `uncertain` | Некоторая оценка есть, но критерии уверенной идентификации не выполнены. |
| `unavailable` | Оценка еще не построена. |

`characteristic_time_identifiable: true` следует считать главным признаком того, что число можно интерпретировать как реально идентифицированный scale, а не просто как значение на границе поиска.

---

# 10. Атрибуты entity: compact diagnostics

По умолчанию `diagnostics: compact`.

## Основные атрибуты состояния

| Атрибут | Смысл |
|---|---|
| `stddev` | Posterior standard deviation итогового уровня. Это неопределенность состояния фильтра, а не `sigma` конкретного сенсора. |
| `filter_mode` | `warmup` до первого состояния или `tracking` после начала нормальной работы. |
| `noise_model_mode` | Что выбрано в YAML: `auto`, `gaussian` или `poisson`. |
| `noise_model` | Какая noise family реально активна сейчас. В 0.2.1.6 `auto -> gaussian`. |
| `noise_model_params` | Только специфические параметры активной noise family. Для Gaussian обычно `{}`. |
| `noise_variance_source` | Обычно `per_source_calibration`; если calibration еще нет — `model_default`. |

## Диагностика последнего update

Все эти поля относятся **к одному и тому же последнему измерению**:

| Атрибут | Смысл |
|---|---|
| `last_source` | Entity ID источника, который дал последний Bayesian update. |
| `measurement_sigma` | Standard deviation observation model именно этого измерения. |
| `measurement_variance` | `measurement_sigma^2`, то есть `R` этого update. |
| `innovation` | `measurement - prediction` до update. Знак сохраняется. |
| `z_score` | Абсолютная нормированная инновация с учетом predicted state uncertainty и measurement variance. |
| `robust_weight` | Вес Student-t update: `1` для нормального измерения, меньше 1 для сомнительного/выброса. |
| `update_dt_s` | Время между текущим и предыдущим обработанным наблюдением фильтра. |

### Пример

```text
last_source         sensor.room_temperature
measurement_sigma   0.24
measurement_variance 0.06
innovation          -0.01
z_score              0.06
robust_weight        1.0
```

Это читается так: последний источник пришел почти точно в prediction и получил полный вес.

При выбросе, например:

```text
innovation      -0.93
z_score          6.65
robust_weight    0.10
```

фильтр почти игнорирует это измерение, но не выключает датчик бинарно.

---

# 11. `source_health`

Для каждого источника публикуется отдельный набор diagnostics.

Пример:

```yaml
source_health:
  sensor.temperature_1:
    bias: 0.41
    sigma: 0.23
    median_dt_s: 15.0
    outliers: 0
    outlier_rate: 0.0
    history_samples: 28000
    calibration_samples: 1100
    calibration_span_s: 40000
    calibration_pairs: 4
    live_updates: 15
    last_raw_value: 23.24
    last_corrected_value: 22.83
    last_innovation: 0.16
    last_z_score: 0.67
    last_robust_weight: 1.0
```

## Значения `source_health`

| Поле | Смысл |
|---|---|
| `bias` | Текущее относительное смещение источника. Коррекция: `raw - bias`. |
| `sigma` | Эффективная observation uncertainty источника в единицах измеряемой величины. |
| `median_dt_s` | Оценка типичного интервала обновления source. Медленно адаптируется online. |
| `outliers` | Число live updates после текущего старта, признанных сильными выбросами (`weight < 0.25` или `z > 4`). |
| `outlier_rate` | `outliers / live_updates`. Всегда смотрите вместе с числом `live_updates`: `1.0` при `2/2` и при `100/100` — статистически разные ситуации. |
| `history_samples` | Число сырых исторических точек Recorder, использованных при startup training. |
| `calibration_samples` | Количество pairwise residual samples, связанных с данным source в calibration window. Это не то же самое, что `history_samples`. |
| `calibration_span_s` | Максимальный временной span pairwise calibration evidence для source. |
| `calibration_pairs` | Сколько других источников реально дало пригодные pairwise equations для этого source. Для полностью связной системы из пяти датчиков нормальное максимальное значение — `4`. |
| `live_updates` | Сколько live updates этого источника обработано после текущего старта. |
| `last_raw_value` | Последнее сырое числовое состояние источника. Появляется после первого live update. |
| `last_corrected_value` | Значение после применения `bias` именно в момент того update. Не переписывается задним числом при последующем изменении bias. |
| `last_innovation` | Innovation этого источника в его последнем update. |
| `last_z_score` | Z-score его последнего update. |
| `last_robust_weight` | Student-t weight его последнего update. |

---

# 12. Атрибуты characteristic time

| Атрибут | Смысл |
|---|---|
| `characteristic_time_s` | Posterior median characteristic time уровня, секунд. Может быть `null`. |
| `characteristic_time_p10_s` | 10-й процентиль профиля `tau`. |
| `characteristic_time_p90_s` | 90-й процентиль профиля `tau`. |
| `characteristic_time_confidence` | Эвристическая confidence 0..1 с учетом ширины posterior, сигнала, покрытия истории, fit error и boundary effects. |
| `characteristic_time_status` | `identified`, `insufficient_signal`, `below_resolution`, `longer_than_history`, `uncertain`, `unavailable`. |
| `characteristic_time_identifiable` | Boolean: можно ли считать `tau` идентифицированным, а не boundary/weak-signal result. |
| `characteristic_time_boundary_limited` | Posterior заметно упирается в нижнюю или верхнюю границу search grid. |

---

# 13. `diagnostics: full`

При `diagnostics: full` добавляются внутренние исследовательские атрибуты.

| Атрибут | Смысл |
|---|---|
| `variance` | Posterior variance уровня, `stddev^2`. |
| `velocity` | Внутренняя локальная скорость уровня, в единицах source в секунду. |
| `velocity_tau_s` | Время затухания predictive local velocity. Это **не** `characteristic_time_s`. |
| `process_noise_q` | Интенсивность внутреннего damped-acceleration process noise. |
| `velocity_tau_p10_s` | 10-й процентиль posterior predictive velocity tau. |
| `velocity_tau_p90_s` | 90-й процентиль. |
| `velocity_tau_confidence` | Confidence внутреннего predictive tau. |
| `velocity_tau_edge_mass` | Posterior mass на краях search grid. |
| `velocity_tau_boundary_limited` | Predictive tau упирается в границу grid. |
| `characteristic_time_edge_mass` | Edge mass profile characteristic time. |
| `characteristic_nugget_variance` | Оцененная nugget variance вариограммы. |
| `characteristic_process_variance` | Оцененная variance латентного процесса. |
| `characteristic_signal_fraction` | `process_variance / (nugget + process_variance)`. |
| `characteristic_fit_error` | Робастная нормированная ошибка variogram fit. |
| `characteristic_lag_count` | Число lag bins в fit. |
| `characteristic_pair_count` | Суммарное число temporal pairs, участвовавших в lag bins. |
| `characteristic_min_lag_s` | Минимальный фактический lag. |
| `characteristic_max_lag_s` | Максимальный фактический lag. |
| `noise_velocity` | Скорость изменения posterior variance; служебная диагностика, не физическая скорость процесса. |
| `innovation_var` | Полная predicted innovation variance `S = HPH' + R`. |
| `p_value` | Gaussian-tail diagnostic для размера innovation. Student-t weight остается реальным robust control. |
| `effective_innovation_variance` | Innovation variance после Student-t downweighting. |

Для постоянного мониторинга обычно лучше `compact`.

---

# 14. Online adaptation

После startup training компонент продолжает адаптацию.

## Bias

Bias обновляется медленно по робастным cross-source residuals в свежем окне.

## Sigma

Sigma обновляется по close-in-time pairwise residuals.

Чтобы быстрый датчик не получил искусственно огромный вес данных только из-за высокой частоты обновления:

- pair samples привязываются к более медленному источнику;
- effective pair count в variance solver ограничивается;
- при слабой идентифицируемости работает variance floor.

## Source cadence

`median_dt_s` медленно подстраивается по live timestamps. Большие провалы связи ограничиваются, чтобы единичный outage не разрушал оценку обычной частоты source.

---

# 15. Поведение при выбросах и локально "уехавшем" датчике

Интеграция специально рассчитана на случай, когда один сенсор начинает наблюдать локально другое состояние.

Например, четыре скорректированных датчика находятся около `22.5..22.8`, а один падает до `21.7`.

Фильтр не обязан считать этот источник "сломавшимся". Вместо этого:

1. innovation становится большой;
2. `z_score` растет;
3. Student-t уменьшает `robust_weight`;
4. влияние конкретного update уменьшается;
5. `outliers` и per-source diagnostics показывают, что происходит.

Это полезнее жесткого выключения источника: если он вернется в согласованный режим, вес автоматически восстановится.

---

# 16. Noise model: текущее состояние и будущий auto-detect

Публичный интерфейс специально разделяет:

```text
noise_model_mode  = стратегия выбора
noise_model       = реально активное семейство
```

В 0.2.1.6:

```text
noise_model_mode: auto
noise_model: gaussian
```

Это сделано для будущего расширения без изменения YAML schema.

Планируемая логика auto-detect для счетных процессов: сравнение моделей вида

```text
Gaussian: Var(y | mu) ~= const
Poisson:  Var(y | mu) ~= k * mu
```

При этом умножение счетов на коэффициент не создает нового семейства. Это все равно `poisson`; коэффициент масштаба относится к observation model.

---

# 17. Важные ограничения

1. **Нет автоматического преобразования единиц.** `C` и `F`, Pa и hPa нельзя смешивать.
2. **Bias относительный.** Без внешнего эталона фильтр не знает абсолютную систематическую ошибку всех датчиков одновременно.
3. **Sigma — effective observation error, а не паспортная accuracy.** Локальные физические различия входят в нее законно.
4. **Characteristic time — observational.** В закрытом контуре управления это характеристика всей наблюдаемой системы, а не чистая физическая постоянная объекта.
5. **Poisson auto-detect пока не реализован.** `auto` в 0.2.1.6 выбирает Gaussian.
6. **YAML only.** Config Flow пока нет.
7. Интеграция использует `numpy` и явно декларирует `numpy>=1.26.0,<3.0.0` в `manifest.json`. Home Assistant использует уже совместимую установленную версию, если она удовлетворяет этому диапазону.


# 18. Roadmap

Планируемые улучшения, которые сознательно **не включены** в 0.2.1.6, чтобы не смешивать проверенную математику с крупным рефакторингом:

- Config Flow / Options Flow при сохранении совместимости с YAML;
- настоящее `noise_model: auto` с устойчивым определением Gaussian vs Poisson;
- разбиение большого `sensor.py` на отдельные `history`, `diagnostics`, `persistence` и runtime-calibration модули без изменения поведения;
- Home Assistant integration tests с mock state machine / Recorder;
- дополнительная диагностика числа свежих и stale источников.

Внутренние численные параметры вроде минимального Student-t веса, PSD floor и freshness policy не являются пользовательскими настройками. Ключевые значения вынесены в `const.py` и снабжены комментариями, но намеренно не выставлены наружу как YAML knobs.

---

# 19. Legacy / migration notes

## `robust_enable`

Старый экспериментальный параметр больше не нужен.

```yaml
robust_enable: true
```

не является частью документированного конфига 0.2.1.6. Student-t robust update включен постоянно.

## `save_every_s`

Правильно:

```yaml
bayes:
  save_every_s: 60
```

Не надо размещать его на одном уровне с `bayes:`.

## Пустой `bayes:`

Допустимо:

```yaml
bayes:
```

или блок можно вообще опустить. В обоих случаях используются значения по умолчанию.

---

# 20. Troubleshooting

## После обновления атрибуты не изменились

Выполните **полный Restart Home Assistant**. Reload YAML не перезагружает Python-модуль custom integration надежно.

## `filter_mode: warmup`

Фильтр еще не получил первого пригодного состояния или не смог восстановить/обучить состояние.

Проверьте:

- `ensemble.sources`;
- что источники numeric и не `unavailable`;
- Recorder;
- логи `bayesian_state_filter`.

## `characteristic_time_status: insufficient_signal`

Это не обязательно ошибка. Для почти постоянного сигнала, где доминирует measurement/counting noise, физически честный результат — признать `tau` неидентифицируемым.

## `characteristic_time_status: longer_than_history`

История слишком короткая относительно динамики процесса. Увеличьте `history_days`, если это оправдано, или интерпретируйте результат как нижнюю границу, а не точную оценку.

## У одного source высокий `outlier_rate`

Смотрите одновременно:

```text
outliers
live_updates
last_raw_value
last_corrected_value
last_innovation
last_z_score
last_robust_weight
```

`outlier_rate: 1.0` при `2/2` после рестарта еще не статистический приговор датчику.

## `sigma` одного датчика выросла после реального локального изменения

Это может быть правильным поведением. `sigma` измеряет согласованность источника с общей латентной величиной, а не только электронный шум сенсора.

---

# 21. Development tests

Для core regression tests Home Assistant не требуется:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

Тесты проверяют:

- Student-t downweighting сильного выброса;
- конечность и положительность posterior uncertainty;
- relative bias calibration;
- pairwise sigma calibration на асинхронных источниках;
- восстановление известных `sigma` на синтетическом стационарном процессе;
- инвариантность startup calibration к порядку history samples;
- политика времени: backdated observation отклоняется, одинаковый timestamp получает минимальный положительный `dt`;
- воспроизводимость startup calibration на одной и той же истории;
- отсутствие фиктивного схлопывания sigma у коррелированного быстрого источника;
- characteristic time на синтетическом OU-процессе.

---

# 22. Структура репозитория

```text
custom_components/
  bayesian_state_filter/
    __init__.py
    manifest.json
    const.py
    sensor.py
    core/
      dynamics.py
      filter.py
      noise_models.py
      process_noise.py
      state_models.py
      training.py
      types.py
      updaters.py
      variogram.py

examples/
  minimal.yaml
  recommended_temperature.yaml
  full.yaml
  poisson.yaml

tests/
  test_core.py
  test_training.py
  test_variogram.py

README.md
README.en.md
CHANGELOG.md
requirements-dev.txt
```

---

## Версия

Текущая версия integration manifest: **0.2.1.6**.
