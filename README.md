# Kursa "Intelektuālu robotu darbību plānošana" noslēguma darba programmas

# Kontūru aproksimācija
### contour_estimator.py

Darbināt ar
```
python contour_estimator.py --input
```
Argumenti:

```--input```, ```-i``` - obligāts arguments - norāde uz attēlu (2D karti);

```--output```, ```-o``` - norāde uz izejas failu atrašanās vietu;

```--overwrite_json```, ```-j``` - pārrakstīt JSON failu;

```--alpha```, ```-a``` - parametrs α nosaka aproksimācijas precizitāti. Noklusējumā α=0.01;

```--draw```, ```-d``` - vizualizēt rezultātu (automātiski, ja nav norādīts ```output```);

```--verbose```, ```-v``` - izvadīt papildus informāciju.

# Diskrētā plānošana
### discrete_planner.py

Darbināt ar
```
python discrete_planner.py --input --start X Y --goal X Y
```
Argumenti:

```--input```, ```-i``` - obligāts arguments - norāde uz attēlu (2D karti);

```--start```, ```-s``` - obilgāts arguments - sākuma punkta X Y;

```--goal```, ```-g``` - obilgāts arguments - mērķa punkta X Y;

```--algorithm``` - obligāts arguments - algoritms (A* vai RRT);

```--seed``` - norādīt nejauši ģenerēto skaitļu sēklu;

```--animate``` - animē rezultātus

```--grid_cols``` - režģa kolonu skaits (noklusējums=50);

```--grid_rows``` - režģa rindu skaits (noklusējums=50);

```--alpha```, ```-a``` - parametrs α nosaka aproksimācijas precizitāti. Noklusējumā α=0.01;

```--verbose```, ```-v``` - izvadīt papildus informāciju.

```--draw```, ```-d``` - vizualizēt rezultātu (automātiski, ja nav norādīts ```output```);


| A* algoritma piemērs              | RRT algoritma piemērs |
|---------------------------|---------------------------|
| ![Alt Text](assets/A_star_repeat.gif) | ![Alt Text](assets/RRT_repeat.gif) |
