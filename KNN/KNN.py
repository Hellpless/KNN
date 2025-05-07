import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time # Pre meranie času

# --- Trieda pre metriky vzdialenosti ---
class DistanceMetrics:
    """
    Trieda obsahujúca statické metódy pre výpočet rôznych metrík vzdialenosti.
    """
    @staticmethod
    def euclidean(point1, point2):
        """
        Vypočíta Euklidovskú vzdialenosť medzi dvoma bodmi.
        Args:
            point1 (np.ndarray): Prvý bod (vektor príznakov).
            point2 (np.ndarray): Druhý bod (vektor príznakov).
        Returns:
            float: Euklidovská vzdialenosť.
        """
        return np.sqrt(np.sum((point1 - point2)**2))

    @staticmethod
    def manhattan(point1, point2):
        """
        Vypočíta Manhattanskú vzdialenosť medzi dvoma bodmi.
        Args:
            point1 (np.ndarray): Prvý bod (vektor príznakov).
            point2 (np.ndarray): Druhý bod (vektor príznakov).
        Returns:
            float: Manhattanská vzdialenosť.
        """
        return np.sum(np.abs(point1 - point2))

    @staticmethod
    def chebyshev(point1, point2):
        """
        Vypočíta Čebyševovu vzdialenosť medzi dvoma bodmi.
        Args:
            point1 (np.ndarray): Prvý bod (vektor príznakov).
            point2 (np.ndarray): Druhý bod (vektor príznakov).
        Returns:
            float: Čebyševova vzdialenosť.
        """
        return np.max(np.abs(point1 - point2))

    @staticmethod
    def minkowski(point1, point2, p=3):
        """
        Vypočíta Minkowského vzdialenosť medzi dvoma bodmi.
        Args:
            point1 (np.ndarray): Prvý bod (vektor príznakov).
            point2 (np.ndarray): Druhý bod (vektor príznakov).
            p (int): Parameter p pre Minkowského vzdialenosť.
        Returns:
            float: Minkowského vzdialenosť.
        """
        return np.power(np.sum(np.power(np.abs(point1 - point2), p)), 1/p)

# --- Váhovacie funkcie ---
def uniform_weights(distances):
    """
    Rovnaké váhy pre všetkých susedov (štandardný KNN).
    Args:
        distances (np.ndarray): Pole vzdialeností k susedom.
    Returns:
        np.ndarray: Pole váh (samé jednotky).
    """
    return np.ones_like(distances)

def inverse_distance_weights(distances, epsilon=1e-6):
    """
    Váhy sú nepriamo úmerné vzdialenosti.
    Args:
        distances (np.ndarray): Pole vzdialeností k susedom.
        epsilon (float): Malá konštanta na zabránenie deleniu nulou.
    Returns:
        np.ndarray: Pole váh.
    """
    # Pridáme epsilon pre numerickú stabilitu a zabránenie deleniu nulou
    return 1.0 / (distances + epsilon)

def exponential_weights(distances, gamma=1.0):
    """
    Váhy klesajú exponenciálne so vzdialenosťou.
    Args:
        distances (np.ndarray): Pole vzdialeností k susedom.
        gamma (float): Parameter kontrolujúci rýchlosť poklesu.
    Returns:
        np.ndarray: Pole váh.
    """
    return np.exp(-gamma * distances)

# --- Trieda WeightedKNN ---
class WeightedKNN:
    """
    Implementácia algoritmu váhovaného K-najbližších susedov.
    """
    def __init__(self, k=3, distance_metric='euclidean', p=3, weighting_func='uniform'):
        """
        Inicializácia klasifikátora.
        Args:
            k (int): Počet susedov.
            distance_metric (str): Názov metriky vzdialenosti ('euclidean', 'manhattan', 'chebyshev', 'minkowski').
            p (int): Parameter p pre Minkowského vzdialenosť (používa sa len ak distance_metric='minkowski').
            weighting_func (str): Názov váhovacej funkcie ('uniform', 'inverse', 'exponential').
        """
        if k <= 0:
            raise ValueError("K musí byť kladné celé číslo.")
        self.k = k
        self.p = p # Pre Minkowského vzdialenosť
        self.X_train = None
        self.y_train = None

        # Nastavenie metriky vzdialenosti
        if distance_metric == 'euclidean':
            self.distance_func = DistanceMetrics.euclidean
        elif distance_metric == 'manhattan':
            self.distance_func = DistanceMetrics.manhattan
        elif distance_metric == 'chebyshev':
            self.distance_func = DistanceMetrics.chebyshev
        elif distance_metric == 'minkowski':
            # Použijeme lambda funkciu na predanie parametra p
            self.distance_func = lambda p1, p2: DistanceMetrics.minkowski(p1, p2, p=self.p)
        else:
            raise ValueError(f"Neznáma metrika vzdialenosti: {distance_metric}")

        # Nastavenie váhovacej funkcie
        if weighting_func == 'uniform':
            self.weighting_func = uniform_weights
        elif weighting_func == 'inverse':
            self.weighting_func = inverse_distance_weights
        elif weighting_func == 'exponential':
            # Použijeme lambda funkciu na prípadné budúce nastavenie gamma
            self.weighting_func = lambda dist: exponential_weights(dist, gamma=1.0)
        else:
            raise ValueError(f"Neznáma váhovacia funkcia: {weighting_func}")

        self.metric_name = distance_metric
        self.weighting_name = weighting_func


    def fit(self, X_train, y_train):
        """
        "Trénovanie" modelu - uloženie trénovacích dát.
        Args:
            X_train (np.ndarray): Trénovacie dáta (príznaky). Shape (n_samples, n_features).
            y_train (np.ndarray): Cieľové premenné (triedy) pre trénovacie dáta. Shape (n_samples,).
        """
        self.X_train = X_train
        self.y_train = y_train

    def _predict_single(self, x_test):
        """
        Predikuje triedu pre jeden testovací bod.
        Args:
            x_test (np.ndarray): Jeden testovací bod (vektor príznakov).
        Returns:
            Predikovaná trieda.
        """
        # 1. Vypočítať vzdialenosti ku všetkým trénovacím bodom
        distances = np.array([self.distance_func(x_test, x_train) for x_train in self.X_train])

        # 2. Nájsť K najbližších susedov
        # Získame indexy K najmenších vzdialeností
        k_indices = np.argsort(distances)[:self.k]

        # Získame triedy a vzdialenosti týchto K susedov
        k_nearest_labels = self.y_train[k_indices]
        k_nearest_distances = distances[k_indices]

        # 3. Vypočítať váhy pre K susedov
        weights = self.weighting_func(k_nearest_distances)

        # 4. Predikovať triedu na základe váhovaného hlasovania
        weighted_votes = {}
        for label, weight in zip(k_nearest_labels, weights):
            weighted_votes[label] = weighted_votes.get(label, 0) + weight

        # Ak nie sú žiadne hlasy (napr. všetky váhy sú 0, čo by nemalo nastať s epsilon),
        # vrátime najčastejšiu triedu medzi K susedmi (fallback na uniform)
        if not weighted_votes:
             # Fallback na majoritné hlasovanie bez váh
             most_common = Counter(k_nearest_labels).most_common(1)
             return most_common[0][0]


        # Nájdi triedu s najväčším súčtom váh
        predicted_label = max(weighted_votes, key=weighted_votes.get)
        return predicted_label

    def predict(self, X_test):
        """
        Predikuje triedy pre viacero testovacích bodov.
        Args:
            X_test (np.ndarray): Testovacie dáta (príznaky). Shape (n_test_samples, n_features).
        Returns:
            np.ndarray: Pole predikovaných tried. Shape (n_test_samples,).
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model musí byť najprv 'natrénovaný' metódou fit().")
        if X_test.shape[1] != self.X_train.shape[1]:
             raise ValueError(f"Počet príznakov v testovacích dátach ({X_test.shape[1]}) sa nezhoduje s trénovacími dátami ({self.X_train.shape[1]}).")

        predictions = np.array([self._predict_single(x_test) for x_test in X_test])
        return predictions

# --- Funkcie pre dáta a vyhodnotenie ---
def load_and_preprocess_data(dataset_loader=load_iris):
    """
    Načíta dáta, rozdelí ich a štandardizuje.
    Args:
        dataset_loader (function): Funkcia na načítanie datasetu (napr. load_iris, load_wine).
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    data = dataset_loader()
    X = data.data
    y = data.target

    # Rozdelenie na tréningovú a testovaciu množinu (pre finálne vyhodnotenie, nie pre CV)
    # X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Štandardizácia dát (veľmi dôležité pre KNN!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Vrátime celé škálované dáta pre krížovú validáciu
    return X_scaled, y

def evaluate_knn_combinations(X, y, k_values, metrics, weightings, p_minkowski=3, n_splits=5):
    """
    Vyhodnotí rôzne kombinácie parametrov KNN pomocou k-násobnej krížovej validácie.
    Args:
        X (np.ndarray): Škálované príznaky.
        y (np.ndarray): Cieľové premenné.
        k_values (list): Zoznam hodnôt K na testovanie.
        metrics (list): Zoznam názvov metrík vzdialenosti.
        weightings (list): Zoznam názvov váhovacích funkcií.
        p_minkowski (int): Parameter p pre Minkowského metriku.
        n_splits (int): Počet záhybov pre krížovú validáciu.
    Returns:
        pd.DataFrame: DataFrame s výsledkami (parametre a priemerná presnosť).
    """
    results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    total_combinations = len(k_values) * len(metrics) * len(weightings)
    current_combination = 0
    start_time_total = time.time()

    print(f"Spúšťa sa vyhodnotenie pre {total_combinations} kombinácií parametrov ({n_splits}-násobná krížová validácia)...")

    for k in k_values:
        for metric in metrics:
            for weighting in weightings:
                current_combination += 1
                start_time_combination = time.time()

                fold_accuracies = []
                # Krížová validácia
                for fold, (train_index, val_index) in enumerate(kf.split(X)):
                    X_train_fold, X_val_fold = X[train_index], X[val_index]
                    y_train_fold, y_val_fold = y[train_index], y[val_index]

                    # Vytvorenie a trénovanie modelu
                    knn = WeightedKNN(k=k, distance_metric=metric, p=p_minkowski, weighting_func=weighting)
                    knn.fit(X_train_fold, y_train_fold)

                    # Predikcia a vyhodnotenie na validačnom záhybe
                    y_pred_fold = knn.predict(X_val_fold)
                    accuracy = accuracy_score(y_val_fold, y_pred_fold)
                    fold_accuracies.append(accuracy)

                # Priemerná presnosť pre danú kombináciu parametrov
                mean_accuracy = np.mean(fold_accuracies)
                std_accuracy = np.std(fold_accuracies)
                end_time_combination = time.time()
                duration_combination = end_time_combination - start_time_combination

                results.append({
                    'K': k,
                    'Metrika': metric + (f'(p={p_minkowski})' if metric == 'minkowski' else ''),
                    'Váhovanie': weighting,
                    'Priemerná presnosť': mean_accuracy,
                    'Štandardná odchýlka': std_accuracy,
                    'Trvanie (s)': duration_combination
                })
                print(f"  Kombinácia {current_combination}/{total_combinations}: K={k}, Metrika={metric}, Váhovanie={weighting} -> Presnosť={mean_accuracy:.4f} (Trvanie: {duration_combination:.2f}s)")


    end_time_total = time.time()
    print(f"\nCelkové vyhodnotenie dokončené za {end_time_total - start_time_total:.2f} sekúnd.")
    return pd.DataFrame(results)

def plot_results(results_df):
    """
    Vykreslí grafy závislosti presnosti od parametrov K, metriky a váhovania.
    Args:
        results_df (pd.DataFrame): DataFrame s výsledkami vyhodnotenia.
    """
    # Graf 1: Presnosť vs. K pre rôzne metriky (priemerované cez váhovania)
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=results_df, x='K', y='Priemerná presnosť', hue='Metrika', marker='o', errorbar='sd', err_style="band")
    plt.title('Priemerná presnosť vs. K pre rôzne metriky vzdialenosti\n(Priemerované cez váhovania, tieňovanie = std. odchýlka)')
    plt.xlabel('Počet susedov (K)')
    plt.ylabel(f'Priemerná presnosť ({results_df["Priemerná presnosť"].mean():.0%} ± std. odch.)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(results_df['K'].unique())
    plt.legend(title='Metrika vzdialenosti')
    plt.tight_layout()
    plt.show()

    # Graf 2: Presnosť vs. K pre rôzne váhovania (priemerované cez metriky)
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=results_df, x='K', y='Priemerná presnosť', hue='Váhovanie', marker='s', errorbar='sd', err_style="band")
    plt.title('Priemerná presnosť vs. K pre rôzne váhovacie funkcie\n(Priemerované cez metriky, tieňovanie = std. odchýlka)')
    plt.xlabel('Počet susedov (K)')
    plt.ylabel(f'Priemerná presnosť ({results_df["Priemerná presnosť"].mean():.0%} ± std. odch.)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(results_df['K'].unique())
    plt.legend(title='Váhovacia funkcia')
    plt.tight_layout()
    plt.show()

    # Graf 3: Heatmapa presnosti pre najlepšie váhovanie
    # Nájdenie najlepšieho váhovania celkovo
    best_weighting = results_df.loc[results_df['Priemerná presnosť'].idxmax()]['Váhovanie']
    pivot_table = results_df[results_df['Váhovanie'] == best_weighting].pivot(index='Metrika', columns='K', values='Priemerná presnosť')

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)
    plt.title(f'Heatmapa priemernej presnosti pre váhovanie: {best_weighting}')
    plt.xlabel('Počet susedov (K)')
    plt.ylabel('Metrika vzdialenosti')
    plt.tight_layout()
    plt.show()


# --- Hlavný skript ---
if __name__ == "__main__":
    # Nastavenie parametrov testovania
    K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]
    METRICS = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    WEIGHTINGS = ['uniform', 'inverse', 'exponential']
    P_MINKOWSKI = 3
    N_SPLITS_CV = 5 # Počet záhybov pre krížovú validáciu

    # 1. Načítanie a príprava dát (Iris dataset)
    print("Načítavam a pripravujem dáta (Iris dataset)...")
    X_scaled, y = load_and_preprocess_data(load_iris)
    print(f"Dáta načítané: {X_scaled.shape[0]} vzoriek, {X_scaled.shape[1]} príznakov.")

    # 2. Vyhodnotenie kombinácií parametrov
    results_df = evaluate_knn_combinations(
        X_scaled, y, K_VALUES, METRICS, WEIGHTINGS, p_minkowski=P_MINKOWSKI, n_splits=N_SPLITS_CV
    )

    # 3. Zobrazenie výsledkov a analýza
    print("\n--- Výsledky krížovej validácie ---")
    # Zoradenie výsledkov od najlepšej presnosti
    results_df_sorted = results_df.sort_values(by='Priemerná presnosť', ascending=False)
    print(results_df_sorted.to_string()) # Zobrazenie celej tabuľky

    best_params = results_df_sorted.iloc[0]
    print("\n--- Najlepšia kombinácia parametrov ---")
    print(f"K: {best_params['K']}")
    print(f"Metrika: {best_params['Metrika']}")
    print(f"Váhovanie: {best_params['Váhovanie']}")
    print(f"Priemerná presnosť: {best_params['Priemerná presnosť']:.4f} (± {best_params['Štandardná odchýlka']:.4f})")

    print("\n--- Stručná analýza (Iris dataset) ---")
    # Tu by nasledovala podrobnejšia textová analýza na základe výsledkov a grafov
    # Napríklad:
    # - Ako sa mení presnosť s K? Existuje optimálne K?
    # - Ktorá metrika vzdialenosti fungovala najlepšie? Prečo?
    # - Pomohlo váhovanie (inverse, exponential) oproti uniformnému? Kedy?
    # - Pre Iris dataset často fungujú dobre stredné hodnoty K (napr. 5-11).
    # - Euklidovská metrika je často dobrým východiskovým bodom pre tento typ dát.
    # - Váhovanie môže pomôcť, najmä ak sú body z rôznych tried blízko seba.

    print("\nGenerujem grafy...")
    # 4. Vizualizácia výsledkov
    plot_results(results_df)

    print("\nHotovo.")

