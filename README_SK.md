# Váhovaný KNN (K-Najbližších Susedov) Klasifikátor s GUI

Tento projekt implementuje klasifikátor K-Najbližších Susedov (KNN) so špecifickou schémou váhovania pre dimenzie a vzdialenosti susedov. Obsahuje grafické používateľské rozhranie (GUI) vytvorené pomocou Tkinter pre interaktívne zadávanie dát, klasifikáciu a vizualizáciu 2D/3D dátových bodov.

Projekt bol vyvíjaný krok za krokom (Schritt für Schritt) ako cvičenie na učenie.

## Funkcionalita

* **Váhovaný KNN Algoritmus**:
    * Počíta váhy dimenzií podľa vzorca: `W_ed = (priemer štvorcov hodnôt v dimenzii d)^(-1)`.
    * Podporuje viacero metrík vzdialenosti pre nájdenie susedov:
        * Váhovaná Euklidovská vzdialenosť
        * Váhovaná Manhattan vzdialenosť
        * Váhovaná Čebyševova vzdialenosť
    * Klasifikuje nové dátové body na základe váhovaných hlasov svojich 'k' najbližších susedov (váha = 1/vzdialenosť).
* **Grafické Používateľské Rozhranie (GUI)**:
    * Vytvorené pomocou knižnice Tkinter v Pythone.
    * Umožňuje používateľom zadávať tréningové dáta (príznaky a triedy).
    * Podporuje dva režimy klasifikácie:
        1.  **Klasifikácia Nového Bodu**: Klasifikuje externe zadaný dátový bod.
        2.  **Klasifikácia Existujúceho Bodu (LOO - Leave-One-Out)**: Klasifikuje existujúci bod z tréningovej sady jeho dočasným vylúčením.
    * Vstupné polia pre počet susedov (`k`) a výber metriky vzdialenosti.
    * Zobrazuje detailné výsledky klasifikácie, vrátane predikovanej triedy, celkových váh pre každú triedu a detailov o k-najbližších susedoch.
    * **2D/3D Vizualizácia Dát**:
        * Používa Matplotlib na vykreslenie tréningových dát, klasifikovaného bodu a jeho najbližších susedov.
        * Podporuje vizualizáciu pre dátové sady s 2 alebo 3 príznakmi.
        * Ak Matplotlib nie je dostupný, GUI bude stále funkčné bez grafu.
* **Generovanie Dát**:
    * Obsahuje skript (`1000.py` alebo `knn_dummy_data_1000.txt`) na generovanie testovacích dát.

## Štruktúra Súborov a Vývoj

Projekt sa vyvíjal cez niekoľko verzií Python skriptov, ktoré demonštrujú postupný vývoj funkcionality:

* `KNN_Understandable.py`: Pravdepodobne počiatočná, viac komentovaná alebo zjednodušená verzia KNN logiky pre lepšie pochopenie.
* `KNN.py`: Základná KNN logika, možno verzia pre príkazový riadok alebo fundamentálny algoritmus.
* `KNN_GUI_V0.1.py`: Prvá verzia GUI, základné zadávanie dát a klasifikácia.
* `KNN_GUI_V0.2(Graph).py`: Pridaná 2D grafická vizualizácia do GUI.
* `KNN_GUI_V0.3(Graph_3D).py`: Rozšírená grafická vizualizácia o podporu 3D dát.
* `KNN_GUI_V0.4(FromMyOwn).py`: Najnovšia verzia, zahŕňajúca režim klasifikácie Leave-One-Out (LOO) a potenciálne ďalšie vylepšenia na základe špecifických požiadaviek používateľa ("FromMyOwn") alebo školských zadaní. Toto je pravdepodobne hlavný súbor aplikácie na spustenie.
* `1000.py`: Python skript na generovanie 1000 testovacích dátových bodov (ako je vidieť v `knn_dummy_data_1000.txt`).
* `knn_dummy_data_1000.txt`: Textový súbor obsahujúci 1000 vygenerovaných testovacích dátových vzoriek.
* `requirements.txt`: Zoznam závislostí projektu (napr. `matplotlib`).

**Hlavný súbor na spustenie aplikácie je `KNN_GUI_V0.4(FromMyOwn).py`.**

## Ako Spustiť

1.  **Predpoklady**:
    * Python 3.x
    * Tkinter (zvyčajne súčasťou Pythonu)
    * Matplotlib (pre grafickú vizualizáciu). Ak nie je nainštalovaný, spustite:
        ```bash
        pip install matplotlib
        ```
    * Skontrolujte `requirements.txt` pre prípadné ďalšie špecifické verzie.

2.  **Klonujte repozitár (ak je to relevantné) alebo stiahnite súbory.**

3.  **Prejdite do priečinka `KNN` vo vašom termináli.**

4.  **Spustite hlavnú aplikáciu**:
    ```bash
    python KNN_GUI_V0.4(FromMyOwn).py
    ```

5.  **Používanie GUI**:
    * Zadajte tréningové dáta vo formáte `priznak1,priznak2,[priznak3],trieda` (jeden bod na riadok).
    * Vyberte režim klasifikácie:
        * **Nový bod**: Zadajte príznaky pre nový bod na klasifikáciu.
        * **Existujúci bod (LOO)**: Kliknite na "Aktualizuj výber bodov pre LOO" pre naplnenie dropdown menu, potom vyberte bod.
    * Zadajte hodnotu `k` (počet susedov).
    * Vyberte požadovanú metriku vzdialenosti.
    * Kliknite na "Klasifikuj".
    * Výsledky sa zobrazia v textovej oblasti a graf sa zobrazí, ak je Matplotlib dostupný a dáta sú 2D alebo 3D.

## Založené na Školskom Zadaní

Tento projekt je založený na školskom zadaní.

