## В двух словах

*Цель:* Создание собственной верифицированной database минералов и ключевых характеристик для последующего использования Machine Learning (ML).
## Задачи

0.  **Исследование: «Локальное развертывание»** (Учитывая небольшой объем данных, работаем на локальной машине)
	a. **Настройка рабочей среды:** Установить **pandas** и **kagglehub** (или **mlcroissant**) внутри локальной папки `.venv` или окружения **conda**. Задокументировать как.
	b. **Загрузка данных:** Скачать набор данных «Unverified Comprehensive database of Minerals» с Kaggle в качестве отправной точки, используя библиотеку **kagglehub** или **mlcroissant**.
	c. **Создание рабочей копии:** Создать копию скачанной «непроверенной таблицы» для редактирования, сохранив оригинал в качестве контрольной версии (baseline) для сравнения в будущем. Продумать структуру путей и систему именования файлов.
	**d. Извлечение IMA данных:** Получить актуальный список минералов и формул из базы **RRUFF (IMA Master List)**. Сохранить его в `0.exploration/raw_database/rruff_database.csv` для вытаскивания оттуда **formula**, **system**, **elements**.
1. **Определение: «Закладка фундамента»**
	a. **Анализ формата:** Изучить формат столбцов оригинала и их смысловое содержание (контекстуальную суть).
	b. **Определение ключевого столбца:** Выбрать и зафиксировать основной столбец (ID или уникальное имя), который станет ключом для нашего набора данных.
	c. **Создание глоссария:** Начать вести заметки для глоссария (например, соответствуя номенклатуре **IMA**)

## 0.Exploration

### a. Локальное развертывание
В VS code было создано рабочее окружение `.venv` и файл с кодом `main.py`, куда были добавлены библиотеки **pandas** и **kugglehub**. 
### b. Загрузка данных
Был скачан database «# Comprehensive database of Minerals» `https://www.kaggle.com/datasets/vinven7/comprehensive-database-of-minerals` с помощью кода 

```python
import pandas as pd 
import os
import kagglehub as kag

# Загрузка базы
path = kag.dataset_download("vinven7/comprehensive-database-of-minerals")
full_path = os.path.join(path, "Minerals_Database.csv")
raw_db = pd.read_csv(full_path)

print(f"Загружено строк: {len(raw_db)}")
```

### c. Создание рабочей копии
Рабочий database был скопирован в путь `0.exploration/work_database/Minerals_Database.csv` и будет полностью верифицирован и изменен под глоссарий в пункте **Наши ключевые колонки**.

### d. Извлечение IMA данных
Был скачан database из сайта RRUFF по ссылке ``https://www.rruff.net/ima-mineral-list/?`` с ключевыми колонками (формулы, элементы, сингония). Из этого database будут скопированы формулы IMA и остальные колонки.
## 1. Definition

### a. Original **unverified** dataset columns
> This dataset is the collection of 3112 minerals, their chemical compositions, crystal structure, physical and optical properties. The properties that are included in this database are the Crystal structure, Mohs Hardness, Refractive Index, Optical axes, Optical Dispersion, Molar Volume, Molar, Mass, Specific Gravity, and Calculated Density.
> (About Dataset card intro)

**The list of minerals with individual pages in Wikipedia is given at: https://en.wikipedia.org/wiki/List_of_minerals. The ‘get’ method of the requests library is used to retrieve this page and the content is parsed using BeautifulSoup – a python library specifically engineered for parsing html and lxml content.** - по-сути это значит:
* что таблицу получили автоматом (1)
* и из Википедии (2) - следовательно информация в таблице от кого угодно и всех сразу.
Поэтому использовать это чтобы учить машину - просто надеяться на авось и так сойдет, а на ML действует железный принцип: "garbage in - garbage out"
Если мы хотим делать адекватный ML, то первый шаг - сделать набор данных, верифицированный специалистом и кросс-валидированный как можно большим количеством реальных экспертов (а не челибосами которые любят представляться "экспертами" сами)

Для последующей работы были взяты только рудные минералы, т.к. они имеют больший интерес, чем остальные минералы в промышленности, но проблема в том, что большинство рудных минералов непрозрачны и мы не можем просто померить их преломление и поглощение света напрямую, поэтому на помощь приходит ключевое свойство для рудных минералов - **отражение**. Если показателем поглощения минералов еще можно пренебречь из-за того, что он мал, то показатель преломления выводится с точностью до двух знаков после запятой. Преломление определяют по отражению измеренное в разных средах с известными показателями преломления (в воздухе и имерсионном масле).
В книге *Шумская Н. И. Определитель рудных минералов по спектральным кривым отражения. - Л.: Недра, 1985. - 231 с.* (нету в цифровом издании) классификация минералов основана на **отражении** и был создан атлас СКО(спекральные кривые отражения), которые были типизированы и по которым можно было легко сравнить результаты СКО для определяемого минерала и найти в книге (краткое изложение [[1985.Шумская Н. И. Определитель рудных минералов по спектральным кривым отражения.SUMMARY- Л. Недра - 231 с.]]).
Мы используем работу Шумской как базовый стандарт для классификации минералов в нашей таблице. Этот фундамент будет расширен и уточнен с помощью данных из других специализированных источников.

Все колонки с отдельными химическими элементами можно заменить на IMA formula. Они находятся на сайте RRUFF. Такая замена позволяет наглядней показать химический состав и занимает меньше места в таблице, а главное делает из огромной таблицы с кучей нулей в нормальную таблицу, колонки которой обозначают одну ключевую характеристику.

1. `Name : str` == Name of mineral
2. `Crystal Structure : float/int` ==  1-Triclinic,2-Monoclinic, 3-Orthorhombic, 4-Tetragonal, 5-Hexagonal, 6-Trigonal,7-Cubic, 8-Amorphous
3. `Optical` == 1-Anisotropic (все кристаллы, кроме кристаллов кубических сингоний), 2-Isotropic(изотропные минералы относятся только к кубической сингонии), 3-Uniaxial(одноосный минерал относят к всем кристаллам тетрагональной, гексагональной и тетрагональной систем, 4-Biaxial (двухосные минералы относятся к ромбической, моноклинной и триклинной). Сразу от колонки можно прикинуть, какой сингонии будет минерал
4. `Mohs Hardness : float (0.0:10.0)` == твердость по шкале Мооса
5.  `Diaphaneity : float/int` == 1-Opaque,2-Translucent,3-Transparent
6. `Specific gravity : float` == the density of mineral divided by the density of water
7. `Optical : float/int` == 1-Anisotropic, 2-Isotropic, 3-Uniaxial, 4-Biaxial
8.  `Reflective index : float` == the ratio of the speed of light in the mineral divided by its speed in free space.
9. `Dispertion : float` == the change in rreflective index of light in the mineral as a function of frequency
10-136. `Chemistry elements : float/int` == containing specific chemistry element in the mineral formula
137 `Count : float` == all containing chemistry elements in the mineral formula
138 `Molar mass : float` == molar mass of mineral
139 `Molar volume : float` == molar volume of mineral
140 `Calculated density : float` == the ratio of the molar mass divided by molar volume


### b. Наши ключевые колонки (column: type == key characteristic)

1. `name : str` == основное (наиболее устоявшееся) название минерала (**на английском**)
2. `aliases : list[str]` == альтернативные названия (*english aliases, но главным образом для вариантов перевода (пока только на русском)*
3. `system : Enum` == сингония минералов
4. `moh : float(0.0:10.0)` == твердость по шкале Мооса 
5. `diaphaneity : Enum` == светопроницаемость
6. `density` == плотность минерала
> Плотность минерала - отношение массы минерала к разнице объема воды при погружении его в сосуд. 
7. `anisotropy : Enum` == ('no', 'uniaxial', 'biaxial', 'isotropic')
>Определяется с помощью polariscope. В видео "How a polariscope reveals a gemstone’s optic character /Gemology" видно как работает этот прибор (https://youtube.com/shorts/vjPKFyzdu2g?si=3fGAY6Lkq65n2r1n)
8. `ligthspeed : float(0.0:1.0)` == скорость света в минерале деленная на скорость света в вакууме
> В видео "Measuring Refractive Indices of a Crystal" гайд, снятый в университете IDAHO, как работник измеряет reflective index с помощью рефрактометра (https://youtu.be/zxZBWpS-VNI?si=_6EYUDigCUMPjdOY). Величина показателя преломления является специфическим признаком каждого минерала, что позволяет его правильно идентифицировать. Чем этот показатель выше, тем больше возможности получения максимального блеска при правильной огранке камня. Высокие показатели преломления у алмаза и циркона, поэтому эти камни характеризуются ярким блеском.
9. `dispersion : float` == разложение света на спектральные цвета при прохождении через оптически плотное вещество
>Для измерения дисперсии кристалла рассчитывают его показатель преломления (преломление измеряют с помощью рефрактометра), используя красный свет (686,7 нм), а затем фиолетовый свет (430,8 нм). Абсолютная разница между красным и фиолетовым показателями преломления кристалла равна его дисперсии. (Значения дисперсии не имеют единиц измерения)
10. `formula : str` == формула, которая описывает минеральный химический состав (сама систематика формул IMA описывается в статье "[A compendium of IMA-Approved Mineral Nomenclature](article/2019.IMA_Nomenclature.FreeToUse.pdf)". Из сайта RRUFF можно скачать .csv с формулами и не только.
11. `elements : list[str]` == список химических элементов входят в состав минерала
