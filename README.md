Task 1 - Perceptron: point on the line
-----------------------------------------
Cílem bylo vytvořit perceptron, který pro 100 generovaných bodů rozhodne, zdali leží nad zadanou přímkou (y = 3x +2), pod ní nebo na ní.
 
- Workflow:
    - Generují se data
        - 100 XY bodů od [-dimension do dimension] + počítají se reálné "pozice" jednotlivých bodů (tzn. nad/pod/na přímce)
    - Trénuje se perceptron
        - vypočítává se chyba a na základě ní se aktualizují váhy a bios
    - Testuje se perceptron
        - získáváme predikce od natrénovaného perceptronu pro dané body
    - Výsledky testování se vykreslí v grafu

Tento perceptron jsem trénoval ve třech "nastaveních" - 2 iterace, 10 iterací a 100 iterací
-> Z jednotlivých grafů můžeme vidět, že perceptron natrénovaný pouze ve 2 iteracích není dostačující, některé body neklasifikuje správně.
  
Mezi 10 a 100 iteracemi není žádný viditelný rozdíl (výkon perceptronu je vesměs totožný).

NOTE: Žlutá barva by měla označit body na přímce, nicméně pro daný np.seed se žádný bod nevygeneroval přesně na ní -> žádný žlutý bod není

  <img src="https://github.com/user-attachments/assets/743d3cda-0333-4b5d-b61d-4fe1f1e31713" width="400" />
  <img src="https://github.com/user-attachments/assets/8758155d-e451-42e8-9ab3-9647c2fa80de" width="400" />
  <img src="https://github.com/user-attachments/assets/89edb614-da77-40c5-8601-3580ddab75d3" width="400" />
  



  
Task 2 - Simple neural network: XOR problem
-----------------------------------------
Cílem bylo vytvořit MLP pro XOR problém, který by úspěšně klasifikoval výsledek dvou stupních operací (0/1)

- Workflow:
    - Vytvoření dat
        - tabulka ze zadání se transformuje do proměnných data_x a data_y (tzn. proměnná pro vstup a proměnná pro očekávaný výstup)
        - také se inicializují váhy a biasy mezi jednotlivými vrstvami (vstup -> hidden | hidden -> výstup)
    - Trénuje se perceptron
        - vypočítává se chyba, která se počítá jako rozdíl mezi očekávanám výstupem a výsledkem z vrstev (vstup -> hidden | hidden -> výstup),
        k tomu se využívá funkce sigmoid, která normalizuje výsledek do intervalu [0;1]
        - následně se provádí "backpropagace" neboli zpětný průchod sítí -> získáváme gradient ( k tomu pomáhá funkce calc_dsigmoid )
        - pomocí získaných gradientů následně aktualizujeme váhy a biasy
    - Testuje se perceptron
        - získáváme predikce od natrénovaného perceptronu pro dané vstupy ( výsledek je hodnota v intervalu 0-1)
    - Zobrazení výsledků
        - výsledky jsou zobrazeny v intervalu 0-1 pro každý vstup (celkem jsou 4 vstupy)

Tento MLP jsem trénoval ve třech "nastaveních" - 10 iterací, 100 iterací a 1000 iterací

-> Z jednotlivých výsledků můžeme vidět, že MLP při 10 a 100 iterací stále nedokáže úspěšně klasifikovat daný XOR problém.

-> Při 1000 iteracích jsou již predikce stabilní.

NOTE: V úvahu se zde musí brát můj learning_rate=0.5 a počet neuronů ve skryté vrstvě (hidden_size = 8), pro jiné nastavení by i výsledky mohly být jiné
 
![image](https://github.com/user-attachments/assets/9c9cc80d-e622-43fc-b10a-2987f286f0a1)



Task 3 - Hopfield network
-----------------------------------------
Cílem bylo vytvořit Hopfieldovu síť pro ukládání a následný "restore" nekompletních nebo jinak pozměněných vzorů

- Workflow:
    - Vytvoření dat
        - inicializovala se váhová matice s rozměry size x size (20 x 20) - zde bylo důležité, aby byla symetrická
    - Trénuje se síť
        - po přidání nějakých vzorů do pole original_patterns pomocí metody add_pattern_to_memory následuje trénování
        - každý vzor se postupně převede na 1D pole pomocí flatten (je to z toho důvodu, že jako vstupní vzor ukládáme 2D pole)
        - následně po reshape získáme sloupcový vektor hodnot
        - pro výpočet skalárního součinu potřebujeme jak sloupcový, tak řádkový vektor, tzn na daný sloupcový vektor zavoláme .T (neboli transpose)
        - následně se každý skalární součin přičtě do váhové matice
        - po iteraci přes všechny vzory jsem se také rozhodl přidat normalizaci vah - mělo by to pomoct stabilitě učení (nejspíše by se to ale projevilo při komplexnější síti, resp. při násobně více vzorech)
        - následovalo přidání 0 na diagonálu váhové matice 
    - Získávají se opravené vzory
        - po přidání nějakých vzorů do pole corrupted_patterns pomocí metody add_corrupted_pattern následují "opravy"
        - zde je podobná myšlena jako u druhého bodu v předchozím kroku - tzn. převedeme 2D pole pomocí flatten na 1D
        - následuje volba, zdali použijeme synchronní nebo asynchronní přístup
            - synchronní:
                - synchronní přístup má výhodu v rychlosti, nicméně se u něj mohou objevovat artefakty (ghosting) - v mém případě jsem na toto nenarazil
                - funguje tak, že se vypočítá skalární součin celé váhové matice a daného "corrupted" vzoru, následně se zavolá na výsledek np.sign(), což vrací hodnoty dle toho, zdali jsou výsledné čísla pozitivní nebo negativní
            - asynchronní
                - asynchronní přístup by měl být více robustnější a odolný vůči problémům synchronního, taky trvá déle
                - jeho logika je velmi podobná synchronnímu, nicméně se aktualizuje pouze jeden náhodně vybraný "pixel" v jeden čas
    - Zobrazení výsledků
        - prvně jsou zobrazeny vstupní vzory, na kterých se model učil
        - následně jsou pro každý vstupní vzor zobrazeny jejich "corrupt" varianta a následná opravená varianta 

NOTE: Pro učení jsem si vybral 3 vzory na 5x5 mřížce, čísla 5, 7 a písmeno H.

Všechny tyto vzory se mi podařilo úspěšně opravit - respektive jejich "corrupt" varianty.

Problémy nicméně byly, pokud jsem chtěl opravovat např. čísla 5 a 3, jelikož mají podobnou strukturu až na 2 pixely - model v tomto případě predikoval pouze jednu z nich (proto jsem číslo 3 nahradil písmenem H)

<img src="https://github.com/user-attachments/assets/5a96b82c-be5b-498f-b220-878992ba40e8" width="400" />
<img src="https://github.com/user-attachments/assets/b3008ccf-242e-40fb-b677-481b5558b4cb" width="400" />
<img src="https://github.com/user-attachments/assets/42c98272-9a17-4e7f-af52-11e9e901ef92" width="400" />
<img src="https://github.com/user-attachments/assets/e3d1aad8-0cd6-493b-824a-003412e8367d" width="400" />







Task 4 - Q-learning and the game Find the cheese
-----------------------------------------

Cílem bylo vytvořit Q-Learning model, kdy agent hledá sýr a minimalizovat uraženou vzdálenost bez toho, aby vstoupil do díry
- Workflow:
    - Inicializace
        - vytváří se matice, která je inicializovaná na 0 - tato matice má na každé pozici [X,Y] 4 hodnoty - jedna hodnota pro každý pohyb
        - vytváří se díry na náhodných pozicích kromě počáteční a konečné

    - Trénink modelu
        - trénování probíhá v N epizodách, na začátku každé epizody se nastaví počáteční "defaultní" hodnoty např. pro aktuální pozici
        - epizoda probíhá do té doby, dokud nenarazíme na cíl - sýr nebo nevstoupíme do díry
            - v každé epizodě se určí možné kroky, které mohou na dané pozici nastat
            - je zde šance, že se vybere náhodný krok (epsilon) nebo krok, který má největší Q-hodnotu v matici (na aktuální pozici)
            - po vybrání kroku se vypočítá na jaké pozici se momentálně agent nachází
            - následuje výpočet odměny (obsahuje i malou penalizaci za pohyb)
            - následně dochází k aktualizaci matice a změny aktuální pozice
    - Zobrazení výsledků
        - výsledky tréninku se vykreslí v animaci (defaulně se zobrazuje jen poslední úspěšná epizoda)

 Tento Q-learning jsem trénoval ve třech "nastaveních" - 5, 10 a 100 epizod na 5x5 matici (mřížce) se třemi dírami
 
   -> 5 epizod nebylo dostatečných, agent v každé z nich vstoupil do díry.
   
   -> Při 10 epizodách již vidíme, že se agent ve 13 krocích dostal k cíli
   
   -> Při 100 již agent dosáhl nejkratší možné cesty k sýru (výsledku).

NOTE: Modrá barva představuje aktuální pozici agenta v mřížce, červenou jsou označeny díry a zelená je cílová pozice

<img src="https://github.com/user-attachments/assets/d9fd0382-aa8d-4923-b52a-a1b9599ae320" width="400" />
<img src="https://github.com/user-attachments/assets/63af6db1-cc83-4f3b-b447-1a0094c755a0" width="400" />



Task 5 - Pole-balancing problem
-----------------------------------------
Task 6 - L-systems
-----------------------------------------

Cílem bylo vytvořit Lindenmayer system, ve kterém budeme moct specifikovat axiom, pravidlo a úhel. Jako výsledek dostaneme jednodušší i složitější obrazce skládající se z jednotlivých úseček.
- Workflow:
    - Inicializace
        - nutno zadat axiom, pravidlo a úhel otočení
        - pro složitější obrazce se také používá "nesting" - tzn. jakési řetězení axiomů (zadává se jako celočíselné číslo)

    - Transformace axiomu a pravidla
        - jako první procházíme každý znak v axiomu
            -  pokud narazíme na znak pravidla ('F'), přidáváme do pole instrukcí dané pravidlo (pravidlo nahrazení např. F -> F+F-F)
            -  pokud narazíme na jakýkoliv jiný znak ( + , - , ] , [ ), pouze jej přidáme do pole instrukcí
        - v těchto transformacích se také pracuje s "nestingem"
            - tzn. pokud máme zadán více než 1 nesting, tak se po první iteraci prochází každý znak né v axiomu, ale v předchozím poli instrukcí
                - obdobně pro každou další iteraci nestingu
        - výsledkem je tedy pole instrukcí

    - Vykonávání instrukcí
        - na začátku se inicializuje aktuální pozice X, Y a směr (směr je defaultně nastaven jako "0" -> tzn pohyb po ose X o +1)
        - následně se prochází každá instrukce:
            - 'F'
                - u této instrukce se provede převod stupňů na radiány (z důvodu konvence)
                - následně se vypočítá posun na ose X v daném směru pomocí funkce math.cos
                - obdobně se vypočítá posun na ose Y v daném směru pomocí math.sin
                - ke každému posunu se přičtou jejich aktuální X a Y pozice -> získáváme tedy new_X a new_Y
                - pozice se aktualizuje a data se uloží pro vizualizaci
            - '+'
                - tato instrukce nám říká, že se směr má posunout (po směru hodinových ručiček) o hodnotu úhlu otočení
            - '-'
                - tato instrukce nám říká, že se směr má posunout (proti směru hodinových ručiček) o hodnotu úhlu otočení
            - '['
                - tato operace nám na zásobník uloží aktuální pozici X, Y a aktuální směr
            - ']'
                - tato operace získá uloženou pozici X, Y a uložený směr z vrcholu zásobníku 
 
 Ve výsledných vizualizacích můžeme vidět, jak i poměrně jednoduchý axiom, pravidlo, úhel a počet větvení může vytvářet zajímavé obrazce.

<img src="https://github.com/user-attachments/assets/81ad149c-59fa-4e2a-81df-57c0cc74e38f" width="400" height="300" />
<img src="https://github.com/user-attachments/assets/1971e9fd-f542-490c-8fbe-ce9056c9b0df" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/f4e5fb76-160c-4841-9301-3333bdc737bc" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/b3e8a59b-b5fa-4772-a9a0-07df33fb151f" width="400" height="300"/>


Task 7 - IFS
-----------------------------------------

Cílem bylo vytvořit 3D fraktály pomocí IFS s využitím náhodně aplikovaných afinních transformací (25% šance)
- Workflow:
    - Inicializace
        - vybere se jeden ze dvou modelů, každý model obsahuje odlišné hodnoty transformací

    - Výpočet transformací
        - v každé iteraci se prochází všechny řádky transformační matice
        - každý řádek má 25% šanci, že se pro něj aplikuje příslušná affinní transformace (transformace se aplikuje pro body X, Y a Z)
            - affinní transformace je dána tímto předpisem, obdobně přepsáno v kódu
            <img src="https://github.com/user-attachments/assets/ec3c79fc-be7f-4e7c-9635-49bab9a6fd20" width="600" height="300" />
            <img src="https://github.com/user-attachments/assets/e687100a-f756-4cb6-8709-0823260574dd" width="600" height="300" />
        - po aplikaci dané affinní transformace se hodnoty X, Y a Z uloží pro vizualizaci
            - na tyto upravené hodnoty (X, Y, Z) navazují další transformace 

    - Vizualizace
 
 Ve výsledných vizualizacích můžeme vidět výsledky pro jednotlivé transformační modely při 100 000 iteracích.
 Jsou to tedy fraktály, které vznikly pomocí jednoduchých transformací - nicméně mají přesto poměrně složitou strukturu.

<img src="https://github.com/user-attachments/assets/570d4d2e-ca34-4ae6-9602-8ced96bd89c9" width="400" height="300" />
<img src="https://github.com/user-attachments/assets/01a8ce2b-f2ea-4a0f-9566-5c2765cbfbf6" width="400" height="300"/>



Task 8 - TEA - Mandelbrot set or Julia's set
-----------------------------------------

Pro tento úkol jsem si vybral Julia's set
- Workflow:
    - Inicializace
        - nastaví se konstanta, obory hodnot pro reálnou a imaginární část
        - nastaví se také parametr size, který ovlivňuje kolik hodnot budeme v jednotlivých prostorech generovat
        - nastaví se taky počet iterací

    - Výpočet Julia's set
        - jako první se vytvoří dvě matice, k tomu slouží obory hodnot a parametr size
            - matice pro reálná čísla a matice pro imaginární čísla
            - tyto matice obsahují rovnoměrně rozmístěná čísla v daných intervalech
        - z těchto matic se vytvoří "mřížka", abychom měli jejich kombinace
        - následuje vytvoření komplexního čísla
            - to se skládá z reálné a imaginární složky
            - v kódu je to zapsáno takhle:  **complex_numbers = real_grid + 1j * imaginary_grid**
        - hlavní iterační smyčka
            - v každé iteraci se vytváří maska, která ověří podmínku, že komplexní číslo je <= 2 (viz. zadání úkolu)
            - následně proběhne aktualizace výsledkové matice na daných pozicích čísel, která splnila podmínku
            - poté se aktualizují samotná komplexní čísla (ty, která splnila podmínku)

    - Vizualizace
 
 Ve výsledných vizualizacích můžeme vidět dva obrazce. První z nich byl generován za pomocí konstanty **(-0.75, 0.10)**, druhý z nich pomocí **(-0.1, 0.65)**. Obě vizualizace byly vytvořeny v 300 iteracích. Obě tyto konstanty jsem si vybral náhodně - zkoušel jsem různé a tyto se mi líbily nejvíce.


Výsledkem jsou tedy velmi zajímavé obrazce, generované pomocí "předpisu" Julia's setu (je to vlastně pouze nastavení oboru hodnot reálné a imaginární části). I přes to, že jsou tyto obrazce vizuálně poměrně složité, jsou vcelku jednoduché na generování.

<img src="https://github.com/user-attachments/assets/1a93ab9d-264c-472c-aa01-2611277d6566" width="400" height="300" />
<img src="https://github.com/user-attachments/assets/e99ab52c-02aa-4bc7-893c-429e6b61657a" width="400" height="300"/>



Task 9 - Generation of 2D country using fractal geometry
-----------------------------------------

Cílem tohoto úkolu bylo vytvořit "krajinu", pomocí fraktální geometrie. Přesněji pomocí spatial subdivision - postupu popsaném v zadání.

- Workflow:
    - Inicializace
        - jako první se nastaví startovní a konečná pozice inicializační úsečky
        - nastaví se taktéž offset (určuje minimální a maximální posun na ose Y)
        - následně se nastaví počet iterací (čím více iterací, tím daná křivka obsahuje více nerovností)
        - jako poslední se vybere barva vykreslení 

    - Generování části krajiny
        - hlavní iterační smyčka
            - v každé iteraci procházíme všechny záznamy v poli **lines** (jsou zde uložené vždy aktuální úsečky - tzn. pouze úsečky z předchozí iterace)
            - pro každý záznam v tomto poli **( x_start, y_start, x_end, y_end )** najdeme prostředek úsečky
            - následuje rozhodnutí, zdali generujeme pod nebo nad aktuální úsečku (k tomu slouží právě proměnná offset)
                - prostředek úsečky na ose Y se o tento posun aktualizuje
            - výsledkem je tedy bod, který je svou X souřadnicí uprostřed dané úsečky, nicméně jeho Y souřadnice je někde v rozsahu <-offset; offset>
            - tento bod se spojí s původními startovními a následně i s konečnými body => uložení pro další iterace
            - po průchodu všemi aktuálními úsečkami nastává nová iterace, ve která se prochází všechny úsečky uložené v minulé iteraci ( neboli příkaz: **lines = new_lines** na konci iterace)
            - v další iteraci se tedy bude generovat z již přetvořených úseček
    - Poslední iterace
        - vždy nás zajímá pouze poslední iterace == hotový produkt
        - pokud se tedy nacházíme v poslední iteraci, ukládáme si X a Y hodnoty pro následné vykreslení prostoru pod křivkou
        - tyto hodnoty jsou následně použity pro funkci plt.fill_between(), která obarví prostor pod křivkou do Y hodnoty -500 s barvou, která je aktuálně nastavená

    - Vizualizace
        - vizualizační okno je omezeno na Y hodnoty z intervalu <-500; 500>
        - jsou vypnuty ukazatele osy X a Y 
 
Výsledkem je generovaná krajina, pomocí fraktální geometrie. Celkem tedy proběhly 3 volání funkce, v prvním volání se generovala zelená krajina, ve druhém krajina šedá a naposledy krajina modrá.

Inicializační parametry jednotlivých částí krajin jsou popsány na obrázku níže.

![image](https://github.com/user-attachments/assets/1b315f56-ef4c-45e2-824e-7e7640cd8dc9)

![image](https://github.com/user-attachments/assets/a091ab2a-0576-47e0-b9e6-8acfe1735494)


Task 10 - Theory of chaos: Logistic map, chaotic numbers and their prediction
-----------------------------------------

Cílem bylo prozkoumat chování logistické mapy v závislosti na parametru a a následně se pokusit naučit neuronovou síť předpovídat body bifurkačního diagramu

- Workflow:
    - Inicializace parametrů
        - vygeneruje se lineární vektor a_values zadaný mezi start a stop s num body
        - nastaví se počet iterací logistické mapy
        - pole x se inicializuje na malé počáteční hodnoty (1e-5)
    - Výpočet logistické mapy
        - v každé iteraci se vypočítá nová hodnota x pomocí vzorce **x = a_values * x * (1 - x)**
        - dle toho, jak je nastavený parametr last, se ukládají kopie aktuálních a_values a x pro pozdější vykreslení bifurkačního diagramu
    - Trénování pomocí NN
        - generuje se 10_000 náhodných vzorků pro a_train, x_train a y_train
            - z těchto vzorků se následně tvoří X_train a y_train data pro vstup do NN
        - síť obsahuje 2 vrstvy s 64 neurony
    - Predikce dat
        - na základě natrénovaného modelu se predikovala hodnota dalšího kroku, znali jsme li krok aktuální (krokem se myslí hodnota a a x)

Výsledkem jsou dva grafy, jeden pro klasický výpočet logistické mapy a druhý pro predikci dat na natrénovaném modelu.
V obou případěch můžeme vidět, že okolo hodnoty a=3.5 se začnou hodnoty chaoticky "měnit" - deterministicky, ale na první pohled náhodně.

<img src="https://github.com/user-attachments/assets/73f5b10a-a4bc-4fe2-924a-ee86cebe0535" width="400" height="300" />
<img src="https://github.com/user-attachments/assets/c85eda4e-0d91-4b04-bc1d-fa011325741a" width="400" height="300" />

Task 12 - Cellular automata
-----------------------------------------

Cílem bylo simulovat šíření požáru v lese pomocí jednoduchého buněčného automatu podle zadaných pravidel.
Celkem zde byly 3 stavy - prázdné místo | strom | hořící strom.

- Workflow:
    - Inicializace mřížky
        - vytvoří se čtvercová mřížka size x size
            - každá buňka je buď prázdná nebo strom (s pravděpodobností p_tree = 0.6)
        - inicializují se také hodnoty **p** a **f**
            - **p** určuje pravděpodobnost, že hořící strom nebo prázdné místo bude nahrazeno za živý strom
            - **f** určuje pravděpodobnost, že strom sám od sebe začne hořet
    - Jednotlové generace
        - pokud alespoň jeden soused (v osmi směrech) hoří, buňka se změní na hořící
        - jinak má pravděpodobnost **f** "samovznícení"
    - Animace
        - vykreslí se 100 generací, které simulují požár
            - bílou barvou je reprezentováno prázdné místo, zelenou strom a červenou oheň
        - animace se prvně vykreslí a následně se uloží do .gif souboru
      
Výsledkem je tedy jakási simulace požáru za předem daných podmínek. Tyto podmínky jsou velmi citlivé na hyperparametry, jalikož například změna **p** o jeden řád ať už směrem nahoru nebo dolů velmi ovlivní průběh regenerace zeleně a šíření požáru.

![simulation](https://github.com/user-attachments/assets/340b4ddf-9bbc-4174-bdb6-5186ad0455b7)

