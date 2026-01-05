## In a nutshell

*Goal:* Making our own **verified** dataset of minerals and their key characteristics first to enable thoughtfull Machine Learning (ML) later.


## Objectives

0. **Exlporation**: "Getting it all locally" *(it is a small data volume after all)*
	a. Set up a *working environment* with `pandas` and `kagglehub` or `mlcroissant` as local `.venv` folder or `conda env` -> **document how?**
	b. Download **unverified** [Comprehensive database of Minerals](article/2019.IMA_Nomenclature.FreeToUse.pdf.pdf) from Kaggle as a *starting point* via `kagglehub` or `mlcroissant`
	b. Make a *working copy* of the downloaded *unverified table* to edit and keep the original as baseline for future -> *think about file naming and paths*
1. **Definition**: "Forging a foundation"
	a. Analyse *original* columns format and contextual essence
	b. Define *our* key column* for the dataset
	c. Start *glossary* notes (IMA nomenclature, например)


## 0.Exploration

***TO BE DOCUMENTED***


## 1. Definition

### a. Original **unverified** dataset columns
> This dataset is the collection of 3112 minerals, their chemical compositions, crystal structure, physical and optical properties. The properties that are included in this database are the Crystal structure, Mohs Hardness, Refractive Index, Optical axes, Optical Dispersion, Molar Volume, Molar, Mass, Specific Gravity, and Calculated Density.
> (About Dataset card intro)

**The list of minerals with individual pages in Wikipedia is given at: https://en.wikipedia.org/wiki/List_of_minerals. The ‘get’ method of the requests library is used to retrieve this page and the content is parsed using BeautifulSoup – a python library specifically engineered for parsing html and lxml content.** - по-сути это значит:
* что таблицу получили автоматом (1)
* и из Википедии (2) - следовательно информация в таблице от кого угодно и всех сразу.
Поэтому использовать это чтобы учить машину - просто надеяться на авось и так сойдет, а ML действует железный принцип: "garbage in - garbage out"
Если мы хотим делать адекватный ML, то первый шаг - сделать набор данных, верифицированный специалистом и кросс-валидированный как можно большим количеством реальных экспертов (а не челибосами которые любят представляться "экспертами" сами)
Все колонки с отдельными химическими элементами можно заменить на [[IMA-nomenclature|International Mineralogical Association (IMA) nomenclature formulae]], т.к. занимает меньше места в таблице и наглядней показывает химический состав минералов, а главное
делает из разреженной таблицы с кучей нулей нормальную таблицу, колонки в которой обозначают одну ключевую характеристику.


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


### b. Our key columns (column: type == key characteristic)

1. `name : str` == основное (наиболее устоявшееся) название минерала (**на английском**)
2. `aliases : list[str]` == альтернативные названия (*english aliases, но главным образом для вариантов перевода (пока только на русском)*
3. `system : Enum` == сингония минералов
4. `moh : float(0.0:10.0)` == твердость по шкале Мооса 
5. `diaphaneity : Enum` == светопроницаемость
6. `density` == плотность минерала
> Плотность минерала - отношение массы минерала к разнице объема воды при погружении его в сосуд. 
7. `anisotropy : Enum` == ('no', 'uniaxial', 'biaxial', ???)
>Определяется с помощью polariscope. В видео "How a polariscope reveals a gemstone’s optic character /Gemology" видно как работает этот прибор (https://youtube.com/shorts/vjPKFyzdu2g?si=3fGAY6Lkq65n2r1n)
8. `ligthspeed : float(0.0:1.0)` == скорость света в минерале деленная на скорость света в вакууме
> В видео "Measuring Refractive Indices of a Crystal" гайд, снятый в университете IDAHO, как работник измеряет reflective index с помощью рефрактометра (https://youtu.be/zxZBWpS-VNI?si=_6EYUDigCUMPjdOY). Величина показателя преломления является специфическим признаком каждого минерала, что позволяет его правильно идентифицировать. Чем этот показатель выше, тем больше возможности получения максимального блеска при правильной огранке камня. Высокие показатели преломления у алмаза и циркона, поэтому эти камни характеризуются ярким блеском.
9. `dispersion : float` == разложение света на спектральные цвета при прохождении через оптически плотное вещество
>Для измерения дисперсии кристалла рассчитывают его показатель преломления (преломление измеряют с помощью рефрактометра), используя красный свет (686,7 нм), а затем фиолетовый свет (430,8 нм). Абсолютная разница между красным и фиолетовым показателями преломления кристалла равна его дисперсии. (Значения дисперсии не имеют единиц измерения)
10. `formula : str` == формула, которая описывает минеральный химический состав (сама систематика формул IMA описывается в статье "[A compendium of IMA-Approved Mineral Nomenclature](article/2019.IMA_Nomenclature.FreeToUse.pdf.pdf)"
11. `elements : list[str]` == список химических элементов входят в состав минерала
