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

Task 4 - Q-learning and the game Find the cheese
-----------------------------------------

Task 5 - Pole-balancing problem
-----------------------------------------
Task 6 - L-systems
-----------------------------------------
Task 7 - IFS
-----------------------------------------
Task 8 - TEA - Mandelbrot set or Julia's set
-----------------------------------------
Task 9 - Generation of 2D country using fractal geometry
-----------------------------------------
Task 10 - Theory of chaos: Logistic map, chaotic numbers and their prediction
-----------------------------------------
Task 12 - Cellular automata
-----------------------------------------
