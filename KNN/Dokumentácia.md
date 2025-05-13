# Dokumentácia: Implementácia KNN a Práca so Vzorcami v Kóde

Táto dokumentácia popisuje, ako Python kód projektu váhovaného KNN klasifikátora priamo implementuje matematické vzorce a algoritmy bez použitia externých knižníc pre kľúčovú logiku KNN.

## 1. Výpočet Váh Dimenzií (`calculate_dimension_weights`)

* **Ako kód pracuje so vzorcom?**
    * Funkcia `calculate_dimension_weights` priamo implementuje vzorec pre výpočet váhy každej dimenzie ($W_{ed}$), ktorý bol špecifikovaný (pravdepodobne zo školského zadania):
        $W_{ed} = \left( \frac{\sum_{i=1}^{N} (X_{id})^2}{N} \right)^{-1}$
        (prevrátená hodnota priemeru druhých mocnín hodnôt v danej dimenzii).
* **Volanie funkcií vs. vlastný výpočet:**
    * Kód si tento vzorec **sám stanovil a vypočítava ho krok po kroku**:
        1.  Iteruje cez každú dimenziu (príznak).
        2.  Pre každú dimenziu iteruje cez všetky tréningové vzorky.
        3.  Počíta sumu druhých mocnín hodnôt (`sum_of_squares += sample_features[d_idx] ** 2`).
        4.  Vypočíta priemer (`mean_of_squares`).
        5.  Výsledná váha je `1 / mean_of_squares` (s ošetrením delenia nulou, kedy je váha `float('inf')`).
    * Nepoužíva žiadnu externú funkciu alebo knižnicu na tento špecifický výpočet váh dimenzií.

## 2. Výpočet Vzdialeností

Kód obsahuje tri funkcie na výpočet váhovaných vzdialeností: `weighted_euclidean_distance`, `weighted_manhattan_distance` a `weighted_chebyshev_distance`.

* **Ako kód pracuje so vzorcami?**
    * Každá z týchto funkcií priamo implementuje príslušný matematický vzorec pre danú metriku vzdialenosti, pričom do výpočtu zahŕňa aj váhy dimenzií ($W_{ed}$) získané z funkcie `calculate_dimension_weights`.
* **Volanie funkcií vs. vlastný výpočet:**
    * **`weighted_euclidean_distance`**:
        * **Vzorec**: $d = \sqrt{ \sum_{j=1}^{D} W_{ej} \cdot (p1_j - p2_j)^2 }$
        * **Implementácia**: Kód iteruje cez dimenzie, počíta vážený štvorec rozdielu pre každú dimenziu (`dim_weights[i] * (p1_features[i] - p2_features[i])**2`), sčítava tieto hodnoty a na záver aplikuje odmocninu (`math.sqrt`).
    * **`weighted_manhattan_distance`**:
        * **Vzorec**: $d = \sum_{j=1}^{D} W_{ej} \cdot |p1_j - p2_j|$
        * **Implementácia**: Kód iteruje cez dimenzie, počíta vážený absolútny rozdiel (`dim_weights[i] * abs(p1_features[i] - p2_features[i])`) a sčítava tieto hodnoty.
    * **`weighted_chebyshev_distance`**:
        * **Vzorec**: $d = \max_{j} (W_{ej} \cdot |p1_j - p2_j|)$
        * **Implementácia**: Kód iteruje cez dimenzie, počíta vážený absolútny rozdiel pre každú dimenziu a vracia maximum z týchto hodnôt.
    * Všetky tieto výpočty sú stanovené priamo v kóde. Používa sa len základná funkcia `math.sqrt` pre odmocninu a vstavaná funkcia `abs` pre absolútnu hodnotu, čo sú štandardné matematické operácie, nie špecializované knižničné funkcie pre výpočet vzdialeností.

## 3. Hlavná Logika Váhovaného KNN (`get_weighted_knn_prediction`)

Táto funkcia realizuje samotný klasifikačný proces.

* **Ako kód pracuje s algoritmom/vzorcami?**
    * Implementuje všetky kroky KNN algoritmu od základov.
* **Volanie funkcií vs. vlastný výpočet:**
    1.  **Výpočet vzdialeností**: Volá **naše vlastné, vyššie definované funkcie** (`weighted_euclidean_distance`, atď.) na výpočet vzdialeností medzi klasifikovaným bodom a všetkými relevantnými tréningovými bodmi.
    2.  **Nájdenie `k` najbližších susedov**: Kód sám zoradí vypočítané vzdialenosti (pomocou vstavanej metódy `sort`) a vyberie `k` bodov s najmenšími vzdialenosťami.
    3.  **Váhovanie susedov**: Pre každého z `k` susedov kód priamo implementuje vzorec pre výpočet jeho váhy: $w_{sused} = 1 / d_{sused}$ (kde $d_{sused}$ je vzdialenosť). Ošetruje aj prípad nulovej vzdialenosti (váha je `float('inf')`).
    4.  **Sčítanie váh pre triedy**: Kód sám iteruje cez `k` susedov a sčítava ich váhy pre príslušné triedy do slovníka (`class_weights`).
    5.  **Určenie výslednej triedy**: Výsledná trieda je tá, ktorá má najvyšší súčet váh. Toto sa zisťuje pomocou vstavanej funkcie `max` na slovníku `class_weights`.
    * Celý tento proces je explicitne naprogramovaný a **nevolá žiadnu komplexnú KNN funkciu z externej knižnice** (ako napr. `KNeighborsClassifier` z `scikit-learn`).

## Záver

Kód pre váhovaný KNN klasifikátor je navrhnutý tak, aby **priamo implementoval všetky potrebné matematické vzorce a algoritmické kroky**. Výpočty váh dimenzií, jednotlivých typov vzdialeností a samotný rozhodovací proces KNN sú stanovené a vykonávané v rámci vlastných funkcií kódu, bez spoliehania sa na predpripravené funkcie z knižníc pre strojové učenie pre tieto kľúčové operácie.
