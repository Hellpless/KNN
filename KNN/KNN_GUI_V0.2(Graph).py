import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import math
from collections import defaultdict

# Pokus o import Matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ==============================================================================
# KROK 1-3: Logika váhovaného KNN (z predchádzajúceho kódu)
# ==============================================================================

def calculate_dimension_weights(data):
    """
    Vypočíta váhy dimenzií (W_ed) podľa vzorca z obrázka:
    W_ed = (priemer štvorcov hodnôt v dimenzii d)^(-1)
    """
    if not data:
        return []
    num_samples = len(data)

    if not data[0][0]:
        return []
    num_dims = len(data[0][0])
    if num_dims == 0:
        return []

    dimension_weights = []
    for d_idx in range(num_dims):
        sum_of_squares = 0
        valid_samples_for_dim = 0
        for sample_features, _ in data:
            if d_idx < len(sample_features):
                sum_of_squares += sample_features[d_idx] ** 2
                valid_samples_for_dim += 1

        if valid_samples_for_dim == 0:
            dimension_weights.append(float('inf'))
            continue

        mean_of_squares = sum_of_squares / valid_samples_for_dim

        if mean_of_squares == 0:
            dimension_weights.append(float('inf'))
        else:
            dimension_weights.append(1 / mean_of_squares)
    return dimension_weights


def weighted_euclidean_distance(point1_features, point2_features, dimension_feature_weights):
    if len(point1_features) != len(point2_features) or len(point1_features) != len(dimension_feature_weights):
        raise ValueError("Počet dimenzií bodov a váh sa musí zhodovať (Euklidovská).")
    sum_sq_diff = 0
    for i in range(len(point1_features)):
        diff = point1_features[i] - point2_features[i]
        sum_sq_diff += dimension_feature_weights[i] * (diff ** 2)
    return math.sqrt(sum_sq_diff)


def weighted_manhattan_distance(point1_features, point2_features, dimension_feature_weights):
    if len(point1_features) != len(point2_features) or len(point1_features) != len(dimension_feature_weights):
        raise ValueError("Počet dimenzií bodov a váh sa musí zhodovať (Manhattan).")
    total_distance = 0
    for i in range(len(point1_features)):
        diff = abs(point1_features[i] - point2_features[i])
        total_distance += dimension_feature_weights[i] * diff
    return total_distance


def weighted_chebyshev_distance(point1_features, point2_features, dimension_feature_weights):
    if len(point1_features) != len(point2_features) or len(point1_features) != len(dimension_feature_weights):
        raise ValueError("Počet dimenzií bodov a váh sa musí zhodovať (Čebyševova).")
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

    # Overenie, či majú všetky tréningové body rovnaký počet príznakov ako nový bod
    num_features_new_point = len(point_to_classify_features)
    for i, (sample_features, _) in enumerate(train_data):
        if len(sample_features) != num_features_new_point:
            raise ValueError(
                f"Nekonzistentný počet príznakov: Tréningový bod X{i + 1} má {len(sample_features)} príznakov, "
                f"ale nový bod má {num_features_new_point} príznakov."
            )

    if k > len(train_data):
        k = len(train_data)

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
            weight = float('inf')
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
        master.title("Váhovaný KNN Klasifikátor s Grafom")
        # Zväčšil som okno, aby sa zmestil aj graf
        master.geometry("800x850")

        # Predvolené hodnoty
        default_training_data = "1.77,80,M\n1.76,49,F\n1.71,56,F\n1.85,96,M"
        default_new_point = "1.68,58"
        default_k = "2"

        # Hlavný PanedWindow pre rozdelenie na vstupy/výstupy a graf
        main_paned_window = ttk.PanedWindow(master, orient=tk.VERTICAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)

        # Horný frame pre vstupy a textové výstupy
        top_frame = ttk.Frame(main_paned_window, padding=5)
        main_paned_window.add(top_frame, weight=1)  # weight určuje, ako sa priestor rozdelí

        # --- Vstupná sekcia v top_frame ---
        input_frame = ttk.LabelFrame(top_frame, text="Vstupné údaje", padding=(10, 5))
        input_frame.pack(padx=10, pady=(10, 0), fill="x")  # pady=(10,0) aby nebol nalepený na spodok

        ttk.Label(input_frame, text="Tréningové dáta (príznak1,príznak2,trieda na riadok):").grid(row=0, column=0,
                                                                                                  sticky="w", padx=5,
                                                                                                  pady=2)
        self.train_data_text = scrolledtext.ScrolledText(input_frame, width=60, height=6, wrap=tk.WORD)
        self.train_data_text.grid(row=1, column=0, columnspan=2, padx=5, pady=2)
        self.train_data_text.insert(tk.END, default_training_data)

        ttk.Label(input_frame, text="Nový bod (príznak1,príznak2):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.new_point_entry = ttk.Entry(input_frame, width=30)
        self.new_point_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        self.new_point_entry.insert(0, default_new_point)

        ttk.Label(input_frame, text="Hodnota k:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.k_entry = ttk.Entry(input_frame, width=10)
        self.k_entry.grid(row=3, column=1, sticky="w", padx=5, pady=2)
        self.k_entry.insert(0, default_k)

        ttk.Label(input_frame, text="Metrika vzdialenosti:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.distance_metric_var = tk.StringVar()
        distance_options = ["Váhovaná Euklidovská", "Váhovaná Manhattan", "Váhovaná Čebyševova"]
        self.distance_metric_menu = ttk.OptionMenu(input_frame, self.distance_metric_var, distance_options[0],
                                                   *distance_options)
        self.distance_metric_menu.grid(row=4, column=1, sticky="ew", padx=5, pady=2)

        self.classify_button = ttk.Button(input_frame, text="Klasifikuj", command=self.classify)
        self.classify_button.grid(row=5, column=0, columnspan=2, pady=10)

        # --- Výstupná sekcia v top_frame ---
        output_frame = ttk.LabelFrame(top_frame, text="Výsledky klasifikácie", padding=(10, 5))
        output_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.result_text = scrolledtext.ScrolledText(output_frame, width=70, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.result_text.pack(padx=5, pady=5, fill="both", expand=True)

        # --- Sekcia pre Graf ---
        # Spodný frame pre graf
        graph_outer_frame = ttk.Frame(main_paned_window, padding=5)
        main_paned_window.add(graph_outer_frame, weight=1)  # Druhý panel

        self.graph_frame = ttk.LabelFrame(graph_outer_frame, text="Grafická vizualizácia (len pre 2D dáta)",
                                          padding=(10, 5))
        self.graph_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.fig = None
        self.ax = None
        self.canvas = None

        if MATPLOTLIB_AVAILABLE:
            self.fig = Figure(figsize=(6, 4), dpi=100)  # Upravená veľkosť pre lepšie zobrazenie
            self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack(fill=tk.BOTH, expand=True)
            self.ax.set_xlabel("Príznak 1")
            self.ax.set_ylabel("Príznak 2")
            self.ax.set_title("KNN Vizualizácia")
            self.canvas.draw()
        else:
            ttk.Label(self.graph_frame, text="Knižnica Matplotlib nie je dostupná. Graf sa nezobrazí.").pack(padx=5,
                                                                                                             pady=5)

    def _parse_training_data(self, data_str):
        parsed_data = []
        lines = data_str.strip().split('\n')
        num_features = None
        for i, line in enumerate(lines):
            if not line.strip(): continue
            parts = line.split(',')
            if len(parts) < 2:
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
                        f"Nekonzistentný počet príznakov na riadku {i + 1}. Očakáva sa {num_features} príznakov, nájdených {len(features)}.")
                if num_features == 0:  # Príznaky musia byť zadané
                    raise ValueError(f"Riadok {i + 1} ('{line}') neobsahuje žiadne číselné príznaky.")

                parsed_data.append((features, class_label))
            except ValueError as e:
                raise ValueError(f"Chyba pri spracovaní riadku {i + 1} ('{line}'): {e}")
        if not parsed_data:
            raise ValueError("Tréningové dáta sú prázdne alebo v nesprávnom formáte.")
        if num_features is None or num_features == 0:  # Ak by boli všetky riadky prázdne alebo bez príznakov
            raise ValueError("Tréningové dáta neobsahujú žiadne platné príznaky.")
        return parsed_data, num_features

    def _parse_new_point(self, point_str, expected_num_features):
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

    def _update_plot(self, training_data_parsed, new_point_parsed, neighbors_info, num_features):
        """Aktualizuje Matplotlib graf."""
        if not MATPLOTLIB_AVAILABLE or self.ax is None:
            return

        self.ax.clear()  # Vyčistenie predchádzajúceho grafu

        if num_features != 2:
            self.ax.text(0.5, 0.5, "Graf je dostupný len pre 2D dáta.",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=10)
            self.ax.set_xlabel("Príznak 1")  # Aj tak nastavíme popisky
            self.ax.set_ylabel("Príznak 2")
            self.ax.set_title("KNN Vizualizácia")
            self.canvas.draw()
            return

        # Získanie unikátnych tried a priradenie farieb
        unique_classes = sorted(list(set(item[1] for item in training_data_parsed)))
        # Použijeme preddefinované farby, alebo cyklické, ak je viac tried
        colors = plt.cm.get_cmap('viridis', len(unique_classes) if len(unique_classes) > 0 else 1)
        class_color_map = {cls: colors(i) for i, cls in enumerate(unique_classes)}

        # Vykreslenie tréningových dát
        for features, class_label in training_data_parsed:
            self.ax.scatter(features[0], features[1],
                            color=class_color_map.get(class_label, 'gray'),  # Sivá pre neznámu triedu
                            label=f"Trieda {class_label}" if class_label not in self.ax.get_legend_handles_labels()[
                                1] else "",
                            alpha=0.7, s=50)  # s je veľkosť bodu

        # Vykreslenie nového bodu
        self.ax.scatter(new_point_parsed[0], new_point_parsed[1], color='red', marker='x', s=100, label="Nový bod")

        # Zvýraznenie najbližších susedov
        neighbor_features_list = [n['features'] for n in neighbors_info]
        if neighbor_features_list:
            for features in neighbor_features_list:
                self.ax.scatter(features[0], features[1],
                                edgecolor='red', facecolors='none',  # Priehľadný vnútro, červený okraj
                                s=150, linewidths=1.5,
                                label="Najbližší sused" if "Najbližší sused" not in self.ax.get_legend_handles_labels()[
                                    1] else "")

        self.ax.set_xlabel("Príznak 1 (napr. Výška)")
        self.ax.set_ylabel("Príznak 2 (napr. Váha)")
        self.ax.set_title("KNN Vizualizácia Dátových Bodov")

        # Zabezpečenie, aby sa legenda zobrazila len raz pre každý typ
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # Odstránenie duplikátov v legende
        self.ax.legend(by_label.values(), by_label.keys())

        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()

    def classify(self):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        # Predvolene vyčistiť graf alebo zobraziť správu, ak dáta nie sú 2D
        if MATPLOTLIB_AVAILABLE and self.ax:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Zadajte dáta a stlačte 'Klasifikuj'.",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=10)
            self.ax.set_xlabel("Príznak 1")
            self.ax.set_ylabel("Príznak 2")
            self.ax.set_title("KNN Vizualizácia")
            self.canvas.draw()

        try:
            train_data_str = self.train_data_text.get(1.0, tk.END)
            training_data_parsed, num_features = self._parse_training_data(train_data_str)

            new_point_str = self.new_point_entry.get()
            new_point_parsed = self._parse_new_point(new_point_str, num_features)

            k_str = self.k_entry.get()
            if not k_str.isdigit() or int(k_str) <= 0:
                raise ValueError("Hodnota k musí byť kladné celé číslo.")
            k = int(k_str)

            distance_metric = self.distance_metric_var.get()

            current_dim_weights = calculate_dimension_weights(training_data_parsed)
            if not current_dim_weights or len(current_dim_weights) != num_features:
                raise ValueError(f"Nepodarilo sa vypočítať váhy dimenzií alebo ich počet ({len(current_dim_weights)}) "
                                 f"nesedí s počtom príznakov ({num_features}).")

            predicted_class, neighbors_info, class_total_weights = get_weighted_knn_prediction(
                training_data_parsed, new_point_parsed, k, distance_metric, current_dim_weights
            )

            # Zobrazenie textových výsledkov
            self.result_text.insert(tk.END, f"--- Výsledky klasifikácie ---\n")
            self.result_text.insert(tk.END, f"Nový bod: {new_point_parsed}\n")
            self.result_text.insert(tk.END, f"Použitá metrika: {distance_metric}\n")
            self.result_text.insert(tk.END, f"Hodnota k: {k}\n")
            self.result_text.insert(tk.END,
                                    f"Vypočítané váhy dimenzií (W_ed): {[f'{w:.6f}' for w in current_dim_weights]}\n\n")
            self.result_text.insert(tk.END, f"Predikovaná trieda: {predicted_class}\n\n")
            self.result_text.insert(tk.END, "Celkové váhy pre jednotlivé triedy:\n")
            for cl, weight_val in class_total_weights.items():
                weight_str = "nekonečno" if weight_val == float('inf') else f"{weight_val:.4f}"
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

            # Aktualizácia grafu
            if MATPLOTLIB_AVAILABLE:
                self._update_plot(training_data_parsed, new_point_parsed, neighbors_info, num_features)

        except ValueError as e:
            messagebox.showerror("Chyba vstupu", str(e))
            self.result_text.insert(tk.END, f"CHYBA: {str(e)}\n")
            if MATPLOTLIB_AVAILABLE and self.ax:  # Vyčistiť graf aj pri chybe
                self.ax.clear()
                self.ax.text(0.5, 0.5, f"Chyba vstupu:\n{str(e)}",
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.ax.transAxes, fontsize=9, color='red')
                self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Neočekávaná chyba", f"Vyskytla sa neočakávaná chyba: {str(e)}")
            self.result_text.insert(tk.END, f"NEOČAKÁVANÁ CHYBA: {str(e)}\n")
            if MATPLOTLIB_AVAILABLE and self.ax:  # Vyčistiť graf aj pri chybe
                self.ax.clear()
                self.ax.text(0.5, 0.5, f"Neočekávaná chyba:\n{str(e)}",
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.ax.transAxes, fontsize=9, color='red')
                self.canvas.draw()
        finally:
            self.result_text.config(state=tk.DISABLED)


if __name__ == '__main__':
    root = tk.Tk()
    app_gui = KNN_GUI(root)
    # Kontrola, či je Matplotlib dostupný, pred spustením
    if not MATPLOTLIB_AVAILABLE:
        print("Upozornenie: Knižnica Matplotlib nie je nainštalovaná. Grafická vizualizácia nebude dostupná.")
        print("Pre inštaláciu spustite: pip install matplotlib")
    root.mainloop()
