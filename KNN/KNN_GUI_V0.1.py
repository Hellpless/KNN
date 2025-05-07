import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import math
from collections import defaultdict


# ==============================================================================
# KROK 1-3: Logika váhovaného KNN (skopírované z predchádzajúceho kódu)
# ==============================================================================

def calculate_dimension_weights(data):
    """
    Vypočíta váhy dimenzií (W_ed) podľa vzorca z obrázka:
    W_ed = (priemer štvorcov hodnôt v dimenzii d)^(-1)
    """
    if not data:
        return []
    num_samples = len(data)

    # Zistenie počtu dimenzií z prvého dátového bodu
    if not data[0][0]:  # Ak prvý bod nemá žiadne príznaky
        return []
    num_dims = len(data[0][0])
    if num_dims == 0:  # Ak sú príznaky prázdna n-tica
        return []

    dimension_weights = []
    for d_idx in range(num_dims):
        sum_of_squares = 0
        valid_samples_for_dim = 0
        for sample_features, _ in data:
            if d_idx < len(sample_features):  # Kontrola, či má bod dostatok príznakov
                sum_of_squares += sample_features[d_idx] ** 2
                valid_samples_for_dim += 1

        if valid_samples_for_dim == 0:  # Žiadne platné dáta pre túto dimenziu
            # print(f"Upozornenie: Žiadne platné dáta pre dimenziu {d_idx}.")
            dimension_weights.append(float('inf'))  # Alebo iná vhodná hodnota
            continue

        mean_of_squares = sum_of_squares / valid_samples_for_dim

        if mean_of_squares == 0:
            # print(f"Upozornenie: Priemer štvorcov pre dimenziu {d_idx} je 0. Váha bude float('inf').")
            dimension_weights.append(float('inf'))
        else:
            dimension_weights.append(1 / mean_of_squares)
    return dimension_weights


def weighted_euclidean_distance(point1_features, point2_features, dimension_feature_weights):
    if len(point1_features) != len(point2_features) or len(point1_features) != len(dimension_feature_weights):
        # Tento error by sa mal odchytiť skôr, pri validácii dát
        # print("Error: Počet dimenzií bodov a váh sa musí zhodovať v Euclidean.")
        raise ValueError("Počet dimenzií bodov a váh sa musí zhodovať.")
    sum_sq_diff = 0
    for i in range(len(point1_features)):
        diff = point1_features[i] - point2_features[i]
        sum_sq_diff += dimension_feature_weights[i] * (diff ** 2)
    return math.sqrt(sum_sq_diff)


def weighted_manhattan_distance(point1_features, point2_features, dimension_feature_weights):
    if len(point1_features) != len(point2_features) or len(point1_features) != len(dimension_feature_weights):
        # print("Error: Počet dimenzií bodov a váh sa musí zhodovať v Manhattan.")
        raise ValueError("Počet dimenzií bodov a váh sa musí zhodovať.")
    total_distance = 0
    for i in range(len(point1_features)):
        diff = abs(point1_features[i] - point2_features[i])
        total_distance += dimension_feature_weights[i] * diff
    return total_distance


def weighted_chebyshev_distance(point1_features, point2_features, dimension_feature_weights):
    if len(point1_features) != len(point2_features) or len(point1_features) != len(dimension_feature_weights):
        # print("Error: Počet dimenzií bodov a váh sa musí zhodovať v Chebyshev.")
        raise ValueError("Počet dimenzií bodov a váh sa musí zhodovať.")
    max_weighted_diff = 0
    for i in range(len(point1_features)):
        diff = abs(point1_features[i] - point2_features[i])
        weighted_diff = dimension_feature_weights[i] * diff
        if weighted_diff > max_weighted_diff:
            max_weighted_diff = weighted_diff
    return max_weighted_diff


def get_weighted_knn_prediction(train_data, point_to_classify_features, k, distance_func_name, current_dim_weights):
    if k <= 0:
        raise ValueError("k musí byť kladné celé číslo.")
    if not train_data:
        raise ValueError("Tréningové dáta nemôžu byť prázdne.")
    if k > len(train_data):
        k = len(train_data)

    # Výber funkcie pre vzdialenosť na základe názvu
    distance_function_map = {
        "Váhovaná Euklidovská": weighted_euclidean_distance,
        "Váhovaná Manhattan": weighted_manhattan_distance,
        "Váhovaná Čebyševova": weighted_chebyshev_distance,
    }
    distance_func = distance_function_map.get(distance_func_name)
    if not distance_func:
        raise ValueError(f"Neznáma metrika vzdialenosti: {distance_func_name}")

    distances = []
    for i, (train_sample_features, train_sample_class) in enumerate(train_data):
        # Overenie konzistencie počtu príznakov
        if len(train_sample_features) != len(point_to_classify_features):
            raise ValueError(
                f"Nekonzistentný počet príznakov: Tréningový bod X{i + 1} ({len(train_sample_features)}) "
                f"vs. nový bod ({len(point_to_classify_features)})."
            )
        dist = distance_func(train_sample_features, point_to_classify_features, current_dim_weights)
        distances.append(((train_sample_features, train_sample_class, f"X{i + 1}"), dist))

    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]

    class_weights = defaultdict(float)
    detailed_neighbors_info = []

    for neighbor_data, dist in neighbors:
        neighbor_features, neighbor_class, neighbor_id = neighbor_data
        weight = 0
        if dist == 0:
            # print(f"Upozornenie GUI: Vzdialenosť k susedovi {neighbor_id} {neighbor_features} ({neighbor_class}) je 0.")
            weight = float('inf')  # Nekonečná váha pre suseda s nulovou vzdialenosťou
        else:
            weight = 1 / dist

        class_weights[neighbor_class] += weight
        detailed_neighbors_info.append({
            "id": neighbor_id, "features": neighbor_features, "class": neighbor_class,
            "distance": dist, "weight": weight
        })

    if not class_weights:
        return "Neznáma", [], {}

    predicted_class = max(class_weights, key=class_weights.get)
    return predicted_class, detailed_neighbors_info, class_weights


# ==============================================================================
# KROK 4: GUI pomocou Tkinter
# ==============================================================================

class KNN_GUI:
    def __init__(self, master):
        self.master = master
        master.title("Váhovaný KNN Klasifikátor")
        master.geometry("700x750")  # Zväčšené okno pre lepšiu prehľadnosť

        # Predvolené hodnoty pre jednoduchšie testovanie
        default_training_data = "1.77,80,M\n1.76,49,F\n1.71,56,F\n1.85,96,M"
        default_new_point = "1.68,58"
        default_k = "2"

        # --- Vstupná sekcia ---
        input_frame = ttk.LabelFrame(master, text="Vstupné údaje", padding=(10, 5))
        input_frame.pack(padx=10, pady=10, fill="x")

        # Tréningové dáta
        ttk.Label(input_frame, text="Tréningové dáta (príznak1,príznak2,trieda na riadok):").grid(row=0, column=0,
                                                                                                  sticky="w", padx=5,
                                                                                                  pady=2)
        self.train_data_text = scrolledtext.ScrolledText(input_frame, width=60, height=8, wrap=tk.WORD)
        self.train_data_text.grid(row=1, column=0, columnspan=2, padx=5, pady=2)
        self.train_data_text.insert(tk.END, default_training_data)

        # Nový bod
        ttk.Label(input_frame, text="Nový bod (príznak1,príznak2):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.new_point_entry = ttk.Entry(input_frame, width=30)
        self.new_point_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        self.new_point_entry.insert(0, default_new_point)

        # Hodnota K
        ttk.Label(input_frame, text="Hodnota k:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.k_entry = ttk.Entry(input_frame, width=10)
        self.k_entry.grid(row=3, column=1, sticky="w", padx=5, pady=2)
        self.k_entry.insert(0, default_k)

        # Metrika vzdialenosti
        ttk.Label(input_frame, text="Metrika vzdialenosti:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.distance_metric_var = tk.StringVar()
        distance_options = ["Váhovaná Euklidovská", "Váhovaná Manhattan", "Váhovaná Čebyševova"]
        self.distance_metric_menu = ttk.OptionMenu(input_frame, self.distance_metric_var, distance_options[0],
                                                   *distance_options)
        self.distance_metric_menu.grid(row=4, column=1, sticky="ew", padx=5, pady=2)

        # Tlačidlo Klasifikuj
        self.classify_button = ttk.Button(input_frame, text="Klasifikuj", command=self.classify)
        self.classify_button.grid(row=5, column=0, columnspan=2, pady=10)

        # --- Výstupná sekcia ---
        output_frame = ttk.LabelFrame(master, text="Výsledky klasifikácie", padding=(10, 5))
        output_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.result_text = scrolledtext.ScrolledText(output_frame, width=80, height=20, wrap=tk.WORD, state=tk.DISABLED)
        self.result_text.pack(padx=5, pady=5, fill="both", expand=True)

    def _parse_training_data(self, data_str):
        """Spracuje textové tréningové dáta."""
        parsed_data = []
        lines = data_str.strip().split('\n')
        num_features = None
        for i, line in enumerate(lines):
            if not line.strip(): continue
            parts = line.split(',')
            if len(parts) < 2:  # Minimálne jeden príznak a trieda
                raise ValueError(
                    f"Nesprávny formát na riadku {i + 1} tréningových dát: '{line}'. Očakáva sa 'p1,p2,...,pn,trieda'.")

            try:
                features_str = parts[:-1]
                features = tuple(float(f.strip()) for f in features_str)
                class_label = parts[-1].strip()
                if not class_label:
                    raise ValueError(f"Chýbajúca trieda na riadku {i + 1}: '{line}'.")

                if num_features is None:
                    num_features = len(features)
                elif len(features) != num_features:
                    raise ValueError(
                        f"Nekonzistentný počet príznakov na riadku {i + 1}. Očakáva sa {num_features} príznakov.")

                parsed_data.append((features, class_label))
            except ValueError as e:
                raise ValueError(f"Chyba pri spracovaní riadku {i + 1} ('{line}'): {e}")
        if not parsed_data:
            raise ValueError("Tréningové dáta sú prázdne alebo v nesprávnom formáte.")
        if num_features == 0:
            raise ValueError("Tréningové dáta musia mať aspoň jeden príznak.")
        return parsed_data, num_features

    def _parse_new_point(self, point_str, expected_num_features):
        """Spracuje textový nový bod."""
        if not point_str.strip():
            raise ValueError("Nový bod na klasifikáciu je prázdny.")
        parts = point_str.split(',')
        try:
            features = tuple(float(f.strip()) for f in parts)
            if len(features) != expected_num_features:
                raise ValueError(
                    f"Nový bod má {len(features)} príznakov, ale tréningové dáta majú {expected_num_features}.")
            return features
        except ValueError as e:
            raise ValueError(f"Chyba pri spracovaní nového bodu ('{point_str}'): {e}")

    def classify(self):
        """Zavolá sa po kliknutí na tlačidlo Klasifikuj."""
        self.result_text.config(state=tk.NORMAL)  # Povolenie zápisu
        self.result_text.delete(1.0, tk.END)  # Vymazanie predchádzajúcich výsledkov

        try:
            # 1. Načítanie a spracovanie vstupov
            train_data_str = self.train_data_text.get(1.0, tk.END)
            training_data_parsed, num_features = self._parse_training_data(train_data_str)

            new_point_str = self.new_point_entry.get()
            new_point_parsed = self._parse_new_point(new_point_str, num_features)

            k_str = self.k_entry.get()
            if not k_str.isdigit() or int(k_str) <= 0:
                raise ValueError("Hodnota k musí byť kladné celé číslo.")
            k = int(k_str)

            distance_metric = self.distance_metric_var.get()

            # 2. Výpočet váh dimenzií
            current_dim_weights = calculate_dimension_weights(training_data_parsed)
            if not current_dim_weights or len(current_dim_weights) != num_features:
                raise ValueError(f"Nepodarilo sa vypočítať váhy dimenzií alebo ich počet ({len(current_dim_weights)}) "
                                 f"nesedí s počtom príznakov ({num_features}).")

            # 3. Spustenie KNN
            predicted_class, neighbors_info, class_total_weights = get_weighted_knn_prediction(
                training_data_parsed,
                new_point_parsed,
                k,
                distance_metric,
                current_dim_weights
            )

            # 4. Zobrazenie výsledkov
            self.result_text.insert(tk.END, f"--- Výsledky klasifikácie ---\n")
            self.result_text.insert(tk.END, f"Nový bod: {new_point_parsed}\n")
            self.result_text.insert(tk.END, f"Použitá metrika: {distance_metric}\n")
            self.result_text.insert(tk.END, f"Hodnota k: {k}\n")
            self.result_text.insert(tk.END, f"Vypočítané váhy dimenzií (W_ed): {current_dim_weights}\n\n")

            self.result_text.insert(tk.END, f"Predikovaná trieda: {predicted_class}\n\n")

            self.result_text.insert(tk.END, "Celkové váhy pre jednotlivé triedy:\n")
            for cl, weight in class_total_weights.items():
                weight_str = "nekonečno" if weight == float('inf') else f"{weight:.4f}"
                self.result_text.insert(tk.END, f"  Trieda {cl}: {weight_str}\n")
            self.result_text.insert(tk.END, "\n")

            self.result_text.insert(tk.END, f"Detail {len(neighbors_info)} najbližších susedov:\n")
            for neighbor in neighbors_info:
                weight_str = "nekonečno" if neighbor['weight'] == float('inf') else f"{neighbor['weight']:.4f}"
                self.result_text.insert(tk.END,
                                        f"  Sused {neighbor['id']}: {neighbor['features']}, "
                                        f"Trieda: {neighbor['class']}, "
                                        f"Vzdialenosť: {neighbor['distance']:.6f}, "
                                        f"Váha: {weight_str}\n"
                                        )

        except ValueError as e:
            messagebox.showerror("Chyba vstupu", str(e))
            self.result_text.insert(tk.END, f"CHYBA: {str(e)}\n")
        except Exception as e:
            messagebox.showerror("Neočekávaná chyba", f"Vyskytla sa neočakávaná chyba: {str(e)}")
            self.result_text.insert(tk.END, f"NEOČAKÁVANÁ CHYBA: {str(e)}\n")
        finally:
            self.result_text.config(state=tk.DISABLED)  # Zakázanie zápisu


if __name__ == '__main__':
    root = tk.Tk()
    app_gui = KNN_GUI(root)
    root.mainloop()
