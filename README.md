## In a nutshell

*Goal:* Making our own **verified** dataset of minerals and their key characteristics first to enable thoughtfull Machine Learning (ML) later.


## Objectives

0. **Exlporation**: "Getting it all locally" *(it is a small data volume after all)*
	a. Set up a *working environment* with `pandas` and `kagglehub` or `mlcroissant` as local `.venv` folder or `conda env` -> **document how?**
	b. Download **unverified** [Comprehensive database of Minerals](https://www.kaggle.com/datasets/vinven7/comprehensive-database-of-minerals) from Kaggle as a *starting point* via `kagglehub` or `mlcroissant`
	b. Make a *working copy* of the downloaded *unverified table* to edit and keep the original as baseline for future -> *think about file naming and paths*
1. **Definition**: "Forging a foundation"
	a. Analyse *original* columns format and contextual essence
	b. Define *our* key column* for the dataset
	c. Start *glossary* notes (IMA nomenclature, например)


## 0.Eploration

***TO BE DOCUMENTED***


## 1. Definition

### a. Original **unverified** dataset columns
> This dataset is the collection of 3112 minerals, their chemical compositions, crystal structure, physical and optical properties. The properties that are included in this database are the Crystal structure, Mohs Hardness, Refractive Index, Optical axes, Optical Dispersion, Molar Volume, Molar, Mass, Specific Gravity, and Calculated Density.
> (About Dataset card intro)

**The list of minerals with individual pages in Wikipedia is given at: https://en.wikipedia.org/wiki/List_of_minerals. The ‘get’ method of the requests library is used to retrieve this page and the content is parsed using BeautifulSoup – a python library specifically engineered for parsing html and lxml content.** - по-сути это значит:
* что таблицу получили автоматом (1)
* и из Википедии (2) - следовательно информацая в таблице от кого угодно и всех сразу.
Поэтому использовать это чтобы учить машину - просто надеятся на авось и так сойдет, а ML действует железный принцип: "garbage in - garbage out"
Если мы хотим делать адекватный ML, то первый шаг - сделать набор данных, верифицированный специалистом и кросс-валидироавнный как можно большим количеством реальных экспертов (а не челибосами которые любят представлятся "экспертами" сами)
Все колонки с отдельными химическими элементами можно заменить на [[IMA-nomenclature|International Mineralogical Association (IMA) nomenclature formulae]], т.к. занимает меньше места в таблице и наглядней показывает химический состав минералов, а главное
делает из разреженной таблицы с кучей нулей нормальную таблицу, колонки в которой обозначают одну ключевую характеристику.


1. `Name : str` == Name of mineral
2. `Crystal Structure : float/int` ==  1-Triclinic,2-Monoclinic, 3-Orthorhombic, 4-Tetragonal, 5-Hexagonal, 6-Trigonal,7-Cubic, 8-Amorphous
	* Тут колонка переименована и убран Amorphous,т.к не подходит под описание колонки
	* Разве? тут ведь не указано сингония - есть ли это слово вообще в англоязычной литературе?
6. `Optical` == 1-Anisotropic (все кристаллы, кроме кристаллов кубических сингоний), 2-Isotropic(изотропные минералы относятся только к кубической сингонии), 3-Uniaxial(одноосный минерал относят к всем кристаллам тетрагональной, гексагональной и тетрагональной систем, 4-Biaxial (двухосные минералы относятся к ромбической, моноклинной и триклинной). Эта колонка самая полезная, т.к. сразу от нее можно прикинуть, какой сингонии будет минерал (хм, наверное)
7 `Refractive Index` == The ratio of the speed of light in the mineral divided by its speed in free space.
Неправильное описание колонки? Все значения больше единицы - значит ли это, что Эйнштейн крутится в гробу со специальной теорией относительности и скорость света не константа максимальной скорости распространения электромагнитных волн?!?

***rest of the original columns with critique - TO BE CONTINUED***


### b. Our key columns (column: type == key characteristic)

1. `name : str` == основное (наиболее устоявшееся) название минерала (**на английском**)
2. `aliases : list[str]` == альтернативные названия (*english aliases, но главным образом для вариантов перевода (пока только на русском)*
3. `system : Enum` == сингония минералов (нашел только что в минералогии на англиском это называют так)
4. `moh : float(0.0:10.0)` == твердость по шкале Мооса 
5. `diaphaneity : Enum` == ??? (нужен наиболее точный термин на русском)
6. `density` == плотность минерала (деленная на плотность воды - а зачем?)
7. `anisotropy : Enum` == ('no', 'uniaxial', 'biaxial', ???)
8. `ligthspeed : float(0.0:1.0)` == скорость света в минерале (деленная на скорость света в вакууме?)
9. `dispersion : float` == колонка Dispersion не влияет на сингонию минералов, поэтому она удалена из таблицы (а вот на мой взгляд это очень интересная колонка, потому что если уж мы определяем скорость света в минерале и оставили предыдущую колонку, то может стоит и эту иметь в виду? просто в википедии про это мало инфы и колонка в основном нули содержит)
10. `formula : str` == формула, которая описывает минеральный химический состав
11. `elements : list[str]` == список химических элементов входят в состав минерала
