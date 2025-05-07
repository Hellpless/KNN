import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import math
from collections import defaultdict

# Pokus o import Matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    # Pre 3D graf
    from mpl_toolkits.mplot3d import Axes3D

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ==============================================================================
# KROK 1-3: Logika váhovaného KNN (z predchádzajúceho kódu)
# ==============================================================================

def calculate_dimension_weights(data):
    """
    Vypočíta váhy dimenzií (W_ed).
    """
    if not data: return []
    num_samples = len(data)
    if not data[0][0]: return []
    num_dims = len(data[0][0])
    if num_dims == 0: return []

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
        dimension_weights.append(1 / mean_of_squares if mean_of_squares != 0 else float('inf'))
    return dimension_weights


def weighted_euclidean_distance(p1_features, p2_features, dim_weights):
    if len(p1_features) != len(p2_features) or len(p1_features) != len(dim_weights):
        raise ValueError("Nezhoda dimenzií pre Euklidovskú vzdialenosť.")
    s = sum(dim_weights[i] * (p1_features[i] - p2_features[i]) ** 2 for i in range(len(p1_features)))
    return math.sqrt(s)


def weighted_manhattan_distance(p1_features, p2_features, dim_weights):
    if len(p1_features) != len(p2_features) or len(p1_features) != len(dim_weights):
        raise ValueError("Nezhoda dimenzií pre Manhattan vzdialenosť.")
    return sum(dim_weights[i] * abs(p1_features[i] - p2_features[i]) for i in range(len(p1_features)))


def weighted_chebyshev_distance(p1_features, p2_features, dim_weights):
    if len(p1_features) != len(p2_features) or len(p1_features) != len(dim_weights):
        raise ValueError("Nezhoda dimenzií pre Čebyševovu vzdialenosť.")
    return max(dim_weights[i] * abs(p1_features[i] - p2_features[i]) for i in range(len(p1_features)))


def get_weighted_knn_prediction(train_data, point_to_classify_features, k, distance_func_name, current_dim_weights):
    if k <= 0: raise ValueError("k musí byť kladné celé číslo.")
    if not train_data: raise ValueError("Tréningové dáta nemôžu byť prázdne.")

    num_features_new_point = len(point_to_classify_features)
    for i, (sample_features, _) in enumerate(train_data):
        if len(sample_features) != num_features_new_point:
            raise ValueError(
                f"Nekonzistentný počet príznakov: Tréningový bod X{i + 1} má {len(sample_features)} príznakov, "
                f"ale nový bod má {num_features_new_point} príznakov."
            )

    if k > len(train_data): k = len(train_data)

    dist_map = {
        "Váhovaná Euklidovská": weighted_euclidean_distance,
        "Váhovaná Manhattan": weighted_manhattan_distance,
        "Váhovaná Čebyševova": weighted_chebyshev_distance,
    }
    dist_func = dist_map.get(distance_func_name)
    if not dist_func: raise ValueError(f"Neznáma metrika: {distance_func_name}")

    distances = []
    for i, (train_sample_features, train_sample_class) in enumerate(train_data):
        dist_val = dist_func(train_sample_features, point_to_classify_features, current_dim_weights)
        distances.append(((train_sample_features, train_sample_class, f"X{i + 1}"), dist_val))

    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]

    class_weights = defaultdict(float)
    detailed_neighbors_info = []
    for neighbor_data, dist_val in neighbors:
        neighbor_features, neighbor_class, neighbor_id = neighbor_data
        weight = float('inf') if dist_val == 0 else 1 / dist_val
        class_weights[neighbor_class] += weight
        detailed_neighbors_info.append({
            "id": neighbor_id, "features": neighbor_features, "class": neighbor_class,
            "distance": dist_val, "weight": weight
        })

    if not class_weights: return "Neznáma", [], {}
    predicted_class = max(class_weights, key=class_weights.get)
    return predicted_class, detailed_neighbors_info, class_weights


# ==============================================================================
# KROK 4: GUI pomocou Tkinter
# ==============================================================================

class KNN_GUI:
    def __init__(self, master):
        self.master = master
        master.title("Váhovaný KNN Klasifikátor s Grafom (2D/3D)")
        master.geometry("850x900")

        # Predvolené 2D dáta
        default_training_data = "1.77,80,M\n1.76,49,F\n1.71,56,F\n1.85,96,M"
        # Príklad pre 3D dáta (môžete odkomentovať a upraviť)
        # default_training_data = "1.77,80,25,M\n1.76,49,30,F\n1.71,56,22,F\n1.85,96,40,M"
        default_new_point = "1.68,58"  # Pre 2D
        # default_new_point = "1.68,58,28" # Pre 3D
        default_k = "2"

        main_paned_window = ttk.PanedWindow(master, orient=tk.VERTICAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)
        top_frame = ttk.Frame(main_paned_window, padding=5)
        main_paned_window.add(top_frame, weight=1)

        input_frame = ttk.LabelFrame(top_frame, text="Vstupné údaje", padding=(10, 5))
        input_frame.pack(padx=10, pady=(10, 0), fill="x")

        ttk.Label(input_frame, text="Tréningové dáta (p1,p2,[p3,]trieda na riadok):").grid(row=0, column=0, sticky="w",
                                                                                           padx=5, pady=2)
        self.train_data_text = scrolledtext.ScrolledText(input_frame, width=70, height=6, wrap=tk.WORD)  # Širšie pole
        self.train_data_text.grid(row=1, column=0, columnspan=2, padx=5, pady=2)
        self.train_data_text.insert(tk.END, default_training_data)

        ttk.Label(input_frame, text="Nový bod (p1,p2[,p3]):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.new_point_entry = ttk.Entry(input_frame, width=40)  # Širšie pole
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

        output_frame = ttk.LabelFrame(top_frame, text="Výsledky klasifikácie", padding=(10, 5))
        output_frame.pack(padx=10, pady=5, fill="both", expand=True)
        self.result_text = scrolledtext.ScrolledText(output_frame, width=80, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.result_text.pack(padx=5, pady=5, fill="both", expand=True)

        graph_outer_frame = ttk.Frame(main_paned_window, padding=5)
        main_paned_window.add(graph_outer_frame, weight=1)
        self.graph_frame = ttk.LabelFrame(graph_outer_frame, text="Grafická vizualizácia (pre 2D alebo 3D dáta)",
                                          padding=(10, 5))
        self.graph_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.fig = None
        self.ax = None  # Bude sa vytvárať dynamicky
        self.canvas = None

        if MATPLOTLIB_AVAILABLE:
            self.fig = Figure(figsize=(7, 5), dpi=100)  # Figúrka pre graf
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack(fill=tk.BOTH, expand=True)
            # Zobrazenie úvodnej správy v grafe
            ax_initial = self.fig.add_subplot(111)
            ax_initial.text(0.5, 0.5, "Graf sa zobrazí po klasifikácii.",
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax_initial.transAxes)
            ax_initial.set_xticks([])
            ax_initial.set_yticks([])
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
                    f"Nesprávny formát na riadku {i + 1} tréningových dát: '{line}'. Očakáva sa 'p1,...,pn,trieda'.")

            try:
                features_str = parts[:-1]
                features = tuple(float(f.strip()) for f in features_str)
                class_label = parts[-1].strip()
                if not class_label: raise ValueError(f"Chýbajúca trieda na riadku {i + 1}: '{line}'.")

                if num_features is None:
                    num_features = len(features)
                elif len(features) != num_features:
                    raise ValueError(
                        f"Nekonzistentný počet príznakov na riadku {i + 1}. Očakáva sa {num_features}, nájdených {len(features)}.")
                if num_features == 0: raise ValueError(f"Riadok {i + 1} ('{line}') neobsahuje číselné príznaky.")
                parsed_data.append((features, class_label))
            except ValueError as e:
                raise ValueError(f"Chyba pri spracovaní riadku {i + 1} ('{line}'): {e}")
        if not parsed_data: raise ValueError("Tréningové dáta sú prázdne alebo v nesprávnom formáte.")
        if num_features is None or num_features == 0:
            raise ValueError("Tréningové dáta neobsahujú platné príznaky.")
        return parsed_data, num_features

    def _parse_new_point(self, point_str, expected_num_features):
        if not point_str.strip(): raise ValueError("Nový bod je prázdny.")
        parts = point_str.split(',')
        try:
            features = tuple(float(f.strip()) for f in parts)
            if len(features) != expected_num_features:
                raise ValueError(
                    f"Nový bod má {len(features)} príznakov, tréningové dáta majú {expected_num_features}.")
            return features
        except ValueError as e:
            raise ValueError(f"Chyba pri spracovaní nového bodu ('{point_str}'): {e}")

    def _update_plot(self, training_data_parsed, new_point_parsed, neighbors_info, num_features):
        if not MATPLOTLIB_AVAILABLE or self.fig is None: return

        self.fig.clear()  # Vyčistenie celej figúrky pred pridaním nového subplotu

        if num_features == 2:
            self.ax = self.fig.add_subplot(111)  # Pridanie 2D subplotu
            unique_classes = sorted(list(set(item[1] for item in training_data_parsed)))
            # Použitie matplotlib.colormaps.get_cmap ak je dostupné, inak staršie plt.cm.get_cmap
            try:
                cmap = plt.colormaps.get_cmap('viridis')
            except AttributeError:
                cmap = plt.cm.get_cmap('viridis')  # Pre staršie verzie Matplotlib

            colors = cmap([i / len(unique_classes) for i in range(len(unique_classes))]) if unique_classes else ['blue']
            class_color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}

            # Vykreslenie tréningových dát
            plotted_labels_train = set()
            for features, class_label in training_data_parsed:
                label_train = f"Trieda {class_label}" if class_label not in plotted_labels_train else None
                self.ax.scatter(features[0], features[1], color=class_color_map.get(class_label, 'gray'),
                                label=label_train, alpha=0.7, s=50)
                if label_train: plotted_labels_train.add(class_label)

            self.ax.scatter(new_point_parsed[0], new_point_parsed[1], color='red', marker='x', s=100, label="Nový bod")

            plotted_label_neighbor = False
            for n_info in neighbors_info:
                features = n_info['features']
                label_neighbor = "Najbližší sused" if not plotted_label_neighbor else None
                self.ax.scatter(features[0], features[1], edgecolor='lime', facecolors='none',
                                s=150, linewidths=1.5, label=label_neighbor)
                if label_neighbor: plotted_label_neighbor = True

            self.ax.set_xlabel("Príznak 1")
            self.ax.set_ylabel("Príznak 2")

        elif num_features == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')  # Pridanie 3D subplotu
            unique_classes = sorted(list(set(item[1] for item in training_data_parsed)))
            try:
                cmap = plt.colormaps.get_cmap('viridis')
            except AttributeError:
                cmap = plt.cm.get_cmap('viridis')

            colors = cmap([i / len(unique_classes) for i in range(len(unique_classes))]) if unique_classes else ['blue']
            class_color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}

            plotted_labels_train = set()
            for features, class_label in training_data_parsed:
                label_train = f"Trieda {class_label}" if class_label not in plotted_labels_train else None
                self.ax.scatter(features[0], features[1], features[2],
                                color=class_color_map.get(class_label, 'gray'),
                                label=label_train, alpha=0.7, s=50)
                if label_train: plotted_labels_train.add(class_label)

            self.ax.scatter(new_point_parsed[0], new_point_parsed[1], new_point_parsed[2],
                            color='red', marker='x', s=100, label="Nový bod")

            plotted_label_neighbor = False
            for n_info in neighbors_info:
                features = n_info['features']
                label_neighbor = "Najbližší sused" if not plotted_label_neighbor else None
                self.ax.scatter(features[0], features[1], features[2], edgecolor='lime',
                                facecolors='none', s=150, linewidths=1.5, label=label_neighbor)
                if label_neighbor: plotted_label_neighbor = True

            self.ax.set_xlabel("Príznak 1")
            self.ax.set_ylabel("Príznak 2")
            self.ax.set_zlabel("Príznak 3")
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.text(0.5, 0.5, "Graf je dostupný len pre 2D alebo 3D dáta.",
                         ha='center', va='center', transform=self.ax.transAxes, fontsize=10)
            self.ax.set_xticks([])
            self.ax.set_yticks([])

        self.ax.set_title("KNN Vizualizácia Dátových Bodov")
        if num_features == 2 or num_features == 3:  # Legenda len pre platné grafy
            handles, labels = self.ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            self.ax.legend(by_label.values(), by_label.keys(), loc='best')  # 'best' pre automatické umiestnenie

        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()

    def classify(self):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        # Inicializácia/vyčistenie grafu pred pokusom o klasifikáciu
        if MATPLOTLIB_AVAILABLE and self.fig:
            self.fig.clear()
            ax_temp = self.fig.add_subplot(111)
            ax_temp.text(0.5, 0.5, "Spracovávam dáta...", ha='center', va='center', transform=ax_temp.transAxes)
            ax_temp.set_xticks([])
            ax_temp.set_yticks([])
            self.canvas.draw()

        try:
            train_data_str = self.train_data_text.get(1.0, tk.END)
            training_data_parsed, num_features = self._parse_training_data(train_data_str)

            new_point_str = self.new_point_entry.get()
            new_point_parsed = self._parse_new_point(new_point_str, num_features)

            k_str = self.k_entry.get()
            if not k_str.isdigit() or int(k_str) <= 0: raise ValueError("k musí byť kladné.")
            k = int(k_str)

            distance_metric = self.distance_metric_var.get()
            current_dim_weights = calculate_dimension_weights(training_data_parsed)
            if not current_dim_weights or len(current_dim_weights) != num_features:
                raise ValueError(f"Váhy dimenzií ({len(current_dim_weights)}) vs príznaky ({num_features}).")

            predicted_class, neighbors_info, class_total_weights = get_weighted_knn_prediction(
                training_data_parsed, new_point_parsed, k, distance_metric, current_dim_weights
            )

            self.result_text.insert(tk.END, f"--- Výsledky klasifikácie ---\n")
            self.result_text.insert(tk.END, f"Nový bod: {new_point_parsed}\n")
            self.result_text.insert(tk.END, f"Metrika: {distance_metric}, k: {k}\n")
            self.result_text.insert(tk.END, f"Váhy dimenzií (W_ed): {[f'{w:.6f}' for w in current_dim_weights]}\n\n")
            self.result_text.insert(tk.END, f"Predikovaná trieda: {predicted_class}\n\n")
            self.result_text.insert(tk.END, "Celkové váhy tried:\n")
            for cl, weight_val in class_total_weights.items():
                w_str = "nekonečno" if weight_val == float('inf') else f"{weight_val:.4f}"
                self.result_text.insert(tk.END, f"  Trieda {cl}: {w_str}\n")
            self.result_text.insert(tk.END, "\nDetail susedov:\n")
            for n in neighbors_info:
                w_str = "nekonečno" if n['weight'] == float('inf') else f"{n['weight']:.4f}"
                self.result_text.insert(tk.END,
                                        f"  {n['id']}: {n['features']}, Tr: {n['class']}, "
                                        f"Vzd: {n['distance']:.6f}, Váha: {w_str}\n"
                                        )

            if MATPLOTLIB_AVAILABLE:
                self._update_plot(training_data_parsed, new_point_parsed, neighbors_info, num_features)

        except ValueError as e:
            messagebox.showerror("Chyba vstupu", str(e))
            self.result_text.insert(tk.END, f"CHYBA: {str(e)}\n")
            if MATPLOTLIB_AVAILABLE and self.fig:
                self.fig.clear()
                ax_err = self.fig.add_subplot(111)
                ax_err.text(0.5, 0.5, f"Chyba vstupu:\n{str(e)}", ha='center', va='center', transform=ax_err.transAxes,
                            fontsize=9, color='red')
                ax_err.set_xticks([]);
                ax_err.set_yticks([])
                self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Neočekávaná chyba", str(e))
            self.result_text.insert(tk.END, f"NEOČAKÁVANÁ CHYBA: {str(e)}\n")
            if MATPLOTLIB_AVAILABLE and self.fig:
                self.fig.clear()
                ax_ex = self.fig.add_subplot(111)
                ax_ex.text(0.5, 0.5, f"Neočekávaná chyba:\n{str(e)}", ha='center', va='center',
                           transform=ax_ex.transAxes, fontsize=9, color='red')
                ax_ex.set_xticks([]);
                ax_ex.set_yticks([])
                self.canvas.draw()
        finally:
            self.result_text.config(state=tk.DISABLED)


if __name__ == '__main__':
    root = tk.Tk()
    app_gui = KNN_GUI(root)
    if not MATPLOTLIB_AVAILABLE:
        print("Upozornenie: Matplotlib nie je nainštalovaný. Graf nebude dostupný.")
        print("Inštalácia: pip install matplotlib")
    root.mainloop()
