import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import math
from collections import defaultdict

# Pokus o import Matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from mpl_toolkits.mplot3d import Axes3D

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ==============================================================================
# Logika váhovaného KNN
# ==============================================================================

def calculate_dimension_weights(data):
    if not data: return []
    num_samples = len(data)
    if not data[0] or not data[0][0]: return []  # Kontrola, či existujú príznaky
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


def get_weighted_knn_prediction(train_data_for_neighbors, point_to_classify_features, k,
                                distance_func_name, current_dim_weights):
    # train_data_for_neighbors: sada dát, z ktorej sa hľadajú susedia (môže byť redukovaná pre LOO)
    # point_to_classify_features: príznaky bodu, ktorý klasifikujeme

    if k <= 0: raise ValueError("k musí byť kladné celé číslo.")
    if not train_data_for_neighbors and k > 0:  # Ak je k>0, potrebujeme susedov
        # Toto by sa malo stať len ak pôvodné tréningové dáta mali len 1 bod a robíme LOO
        raise ValueError("Nie sú dostupní žiadni susedia na klasifikáciu (train_data_for_neighbors je prázdne).")

    num_features_point_to_classify = len(point_to_classify_features)
    for i, (sample_features, _) in enumerate(train_data_for_neighbors):  # Kontrola voči sade pre susedov
        if len(sample_features) != num_features_point_to_classify:
            raise ValueError(
                f"Nekonzistentný počet príznakov: Susediaci bod X{i + 1} má {len(sample_features)} príznakov, "
                f"ale klasifikovaný bod má {num_features_point_to_classify} príznakov."
            )

    # Ak je k väčšie ako počet dostupných susedov, použijeme všetkých dostupných susedov
    if k > len(train_data_for_neighbors):
        k = len(train_data_for_neighbors)

    if k == 0 and len(train_data_for_neighbors) == 0:  # Špeciálny prípad, ak neostali žiadni susedia
        return "Neznáma (žiadni susedia)", [], {}

    dist_map = {
        "Váhovaná Euklidovská": weighted_euclidean_distance,
        "Váhovaná Manhattan": weighted_manhattan_distance,
        "Váhovaná Čebyševova": weighted_chebyshev_distance,
    }
    dist_func = dist_map.get(distance_func_name)
    if not dist_func: raise ValueError(f"Neznáma metrika: {distance_func_name}")

    distances = []
    # Priradenie dočasných ID pre body v train_data_for_neighbors pre sledovanie
    for i, (train_sample_features, train_sample_class) in enumerate(train_data_for_neighbors):
        dist_val = dist_func(train_sample_features, point_to_classify_features, current_dim_weights)
        # Použijeme index + 1 ako dočasné ID v rámci tejto redukovanej sady
        distances.append(((train_sample_features, train_sample_class, f"Sused {i + 1}"), dist_val))

    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]

    class_weights = defaultdict(float)
    detailed_neighbors_info = []
    for neighbor_data, dist_val in neighbors:
        neighbor_features, neighbor_class, neighbor_id_temp = neighbor_data  # neighbor_id_temp je len pre túto funkciu
        weight = float('inf') if dist_val == 0 else 1 / dist_val
        class_weights[neighbor_class] += weight
        detailed_neighbors_info.append({
            "id": neighbor_id_temp,  # Toto ID je relatívne k train_data_for_neighbors
            "features": neighbor_features, "class": neighbor_class,
            "distance": dist_val, "weight": weight
        })

    if not class_weights: return "Neznáma (žiadne váhy)", [], {}
    predicted_class = max(class_weights, key=class_weights.get)
    return predicted_class, detailed_neighbors_info, class_weights


# ==============================================================================
# GUI pomocou Tkinter
# ==============================================================================

class KNN_GUI:
    def __init__(self, master):
        self.master = master
        master.title("Váhovaný KNN Klasifikátor (Nový bod / LOO)")
        master.geometry("900x950")

        default_training_data = "0,0,0,T1\n1,0,1,T1\n0,1,2,T2\n2,1,1,T2\n2,2,2,T3\n0,1,0,T3"  # Dáta z druhého obrázka
        default_new_point = ""  # Predvolene prázdne, keďže môžeme robiť LOO
        default_k = "3"

        main_paned_window = ttk.PanedWindow(master, orient=tk.VERTICAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)
        top_frame = ttk.Frame(main_paned_window, padding=5)
        main_paned_window.add(top_frame, weight=1)

        input_frame = ttk.LabelFrame(top_frame, text="Vstupné údaje", padding=(10, 5))
        input_frame.pack(padx=10, pady=(10, 0), fill="x", expand=True)

        # --- Režim klasifikácie ---
        mode_frame = ttk.Frame(input_frame)
        mode_frame.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        ttk.Label(mode_frame, text="Režim klasifikácie:").pack(side=tk.LEFT, padx=5)
        self.classification_mode = tk.StringVar(value="Nový bod")
        ttk.Radiobutton(mode_frame, text="Nový bod", variable=self.classification_mode,
                        value="Nový bod", command=self._on_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Existujúci bod (LOO)", variable=self.classification_mode,
                        value="LOO", command=self._on_mode_change).pack(side=tk.LEFT)

        # --- Tréningové dáta ---
        ttk.Label(input_frame, text="Tréningové dáta (p1,...,pn,trieda na riadok):").grid(row=1, column=0, sticky="nw",
                                                                                          padx=5, pady=2)
        self.train_data_text = scrolledtext.ScrolledText(input_frame, width=45, height=7, wrap=tk.WORD)
        self.train_data_text.grid(row=1, column=1, sticky="ew", padx=5, pady=2, rowspan=3)  # rowspan=3
        self.train_data_text.insert(tk.END, default_training_data)

        # Načítanie bodov pre LOO výber
        self.load_points_button = ttk.Button(input_frame, text="Aktualizuj výber bodov pre LOO",
                                             command=self._populate_loo_combobox)
        self.load_points_button.grid(row=2, column=0, sticky="w", padx=5, pady=5)

        # --- Nový bod / Výber existujúceho bodu ---
        self.new_point_label = ttk.Label(input_frame, text="Nový bod (p1,...,pn):")
        self.new_point_label.grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.new_point_entry = ttk.Entry(input_frame, width=40)
        self.new_point_entry.grid(row=4, column=1, sticky="ew", padx=5, pady=2)
        self.new_point_entry.insert(0, default_new_point)

        self.loo_point_label = ttk.Label(input_frame, text="Vyber bod na LOO klasifikáciu:")
        self.loo_point_label.grid(row=5, column=0, sticky="w", padx=5, pady=2)
        self.loo_point_var = tk.StringVar()
        self.loo_point_combobox = ttk.Combobox(input_frame, textvariable=self.loo_point_var, width=37, state="disabled")
        self.loo_point_combobox.grid(row=5, column=1, sticky="ew", padx=5, pady=2)

        # --- Ostatné vstupy ---
        ttk.Label(input_frame, text="Hodnota k:").grid(row=6, column=0, sticky="w", padx=5, pady=2)
        self.k_entry = ttk.Entry(input_frame, width=10)
        self.k_entry.grid(row=6, column=1, sticky="w", padx=5, pady=2)  # sticky w
        self.k_entry.insert(0, default_k)

        ttk.Label(input_frame, text="Metrika vzdialenosti:").grid(row=7, column=0, sticky="w", padx=5, pady=2)
        self.distance_metric_var = tk.StringVar()
        distance_options = ["Váhovaná Euklidovská", "Váhovaná Manhattan", "Váhovaná Čebyševova"]
        self.distance_metric_menu = ttk.OptionMenu(input_frame, self.distance_metric_var, distance_options[0],
                                                   *distance_options)
        self.distance_metric_menu.grid(row=7, column=1, sticky="ew", padx=5, pady=2)

        self.classify_button = ttk.Button(input_frame, text="Klasifikuj", command=self.classify)
        self.classify_button.grid(row=8, column=0, columnspan=2, pady=10)

        input_frame.columnconfigure(1, weight=1)  # Aby sa vstupné polia roztiahli

        # --- Výstupná sekcia ---
        output_frame = ttk.LabelFrame(top_frame, text="Výsledky klasifikácie", padding=(10, 5))
        output_frame.pack(padx=10, pady=5, fill="both", expand=True)
        self.result_text = scrolledtext.ScrolledText(output_frame, width=80, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.result_text.pack(padx=5, pady=5, fill="both", expand=True)

        # --- Grafická sekcia ---
        graph_outer_frame = ttk.Frame(main_paned_window, padding=5)
        main_paned_window.add(graph_outer_frame, weight=1)
        self.graph_frame = ttk.LabelFrame(graph_outer_frame, text="Grafická vizualizácia (pre 2D alebo 3D dáta)",
                                          padding=(10, 5))
        self.graph_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.fig = None;
        self.canvas = None
        if MATPLOTLIB_AVAILABLE:
            self.fig = Figure(figsize=(7, 5), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack(fill=tk.BOTH, expand=True)
            self._plot_initial_message("Graf sa zobrazí po klasifikácii.")
        else:
            ttk.Label(self.graph_frame, text="Matplotlib nie je dostupný.").pack(padx=5, pady=5)

        self._on_mode_change()  # Inicializácia stavu GUI podľa predvoleného módu
        self._populate_loo_combobox()  # Prvotné naplnenie comboboxu

    def _plot_initial_message(self, message):
        if not MATPLOTLIB_AVAILABLE or not self.fig: return
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([]);
        ax.set_yticks([])
        self.canvas.draw()

    def _on_mode_change(self):
        mode = self.classification_mode.get()
        if mode == "Nový bod":
            self.new_point_entry.config(state="normal")
            self.new_point_label.config(state="normal")
            self.loo_point_combobox.config(state="disabled")
            self.loo_point_label.config(state="disabled")
            self.load_points_button.config(state="disabled")
        elif mode == "LOO":
            self.new_point_entry.config(state="disabled")
            self.new_point_label.config(state="disabled")
            self.loo_point_combobox.config(state="readonly")  # readonly aby sa dalo vybrať, nie písať
            self.loo_point_label.config(state="normal")
            self.load_points_button.config(state="normal")
            self._populate_loo_combobox()  # Aktualizuj výber pri prepnutí módu

    def _parse_training_data(self, data_str):
        parsed_data = []
        point_identifiers_for_loo = []  # Pre combobox
        lines = data_str.strip().split('\n')
        num_features = None
        for i, line in enumerate(lines):
            original_line_id = f"X{i + 1}"  # Pôvodné ID bodu
            if not line.strip(): continue
            parts = line.split(',')
            if len(parts) < 2:
                raise ValueError(f"Nesprávny formát na riadku {i + 1} ({original_line_id}): '{line}'.")

            try:
                features_str = parts[:-1]
                features = tuple(float(f.strip()) for f in features_str)
                class_label = parts[-1].strip()
                if not class_label: raise ValueError(f"Chýbajúca trieda na riadku {i + 1} ({original_line_id}).")

                if num_features is None:
                    num_features = len(features)
                elif len(features) != num_features:
                    raise ValueError(
                        f"Nekonzistentný počet príznakov na riadku {i + 1} ({original_line_id}). Očakáva sa {num_features}, nájdených {len(features)}.")
                if num_features == 0: raise ValueError(f"Riadok {i + 1} ({original_line_id}) neobsahuje príznaky.")

                parsed_data.append(((features, class_label), original_line_id))  # Ukladáme aj original_line_id
                point_identifiers_for_loo.append(f"{original_line_id}: {features} -> {class_label}")

            except ValueError as e:
                raise ValueError(f"Chyba spracovania riadku {i + 1} ('{line}'): {e}")
        if not parsed_data: raise ValueError("Tréningové dáta sú prázdne.")
        if num_features is None or num_features == 0:
            raise ValueError("Tréningové dáta neobsahujú platné príznaky.")
        return parsed_data, num_features, point_identifiers_for_loo

    def _populate_loo_combobox(self):
        """Naplní combobox pre výber LOO bodu na základe aktuálnych tréningových dát."""
        try:
            train_data_str = self.train_data_text.get(1.0, tk.END)
            # Parsujeme len pre získanie identifikátorov, chyby tu nechceme hádzať ako messagebox
            parsed_data, _, point_identifiers = self._parse_training_data(train_data_str)
            self.loo_point_combobox['values'] = point_identifiers
            if point_identifiers:
                self.loo_point_combobox.current(0)  # Predvolene vyber prvý
            else:
                self.loo_point_combobox.set('')
        except ValueError:  # Ak sú dáta neplatné, combobox bude prázdny
            self.loo_point_combobox['values'] = []
            self.loo_point_combobox.set('')

    def _parse_new_point(self, point_str, expected_num_features):
        # Táto funkcia sa použije len ak je režim "Nový bod"
        if not point_str.strip(): raise ValueError("Nový bod je prázdny.")
        parts = point_str.split(',')
        try:
            features = tuple(float(f.strip()) for f in parts)
            if len(features) != expected_num_features:
                raise ValueError(f"Nový bod má {len(features)} príznakov, tréningové dáta {expected_num_features}.")
            return features
        except ValueError as e:
            raise ValueError(f"Chyba spracovania nového bodu ('{point_str}'): {e}")

    def _update_plot(self, all_training_data_with_ids, point_being_classified_features,
                     point_being_classified_original_id,  # ID bodu, ak je z LOO, inak None
                     neighbors_info, num_features):
        if not MATPLOTLIB_AVAILABLE or self.fig is None: return
        self.fig.clear()

        ax_title = "KNN Vizualizácia"
        if point_being_classified_original_id:
            ax_title += f" (LOO pre {point_being_classified_original_id})"

        if num_features == 2:
            ax = self.fig.add_subplot(111)
            unique_classes = sorted(list(set(item[0][1] for item in all_training_data_with_ids)))
            try:
                cmap = plt.colormaps.get_cmap('viridis')
            except AttributeError:
                cmap = plt.cm.get_cmap('viridis')
            colors = cmap([i / len(unique_classes) for i in range(len(unique_classes))]) if unique_classes else ['blue']
            class_color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}

            plotted_labels_train = set()
            for (features, class_label), original_id in all_training_data_with_ids:
                is_being_classified = (original_id == point_being_classified_original_id)
                marker = 'o'
                size = 70 if is_being_classified else 50
                edge_color_train = 'black' if is_being_classified else None  # Zvýraznenie klasifikovaného bodu

                label_train = f"Trieda {class_label}" if class_label not in plotted_labels_train else None
                ax.scatter(features[0], features[1], color=class_color_map.get(class_label, 'gray'),
                           label=label_train, alpha=0.7, s=size, marker=marker, edgecolor=edge_color_train)
                if label_train: plotted_labels_train.add(class_label)

            # Bod, ktorý sa klasifikuje (či už nový alebo LOO)
            ax.scatter(point_being_classified_features[0], point_being_classified_features[1],
                       color='red', marker='x', s=120, label="Klasifikovaný bod", zorder=5)

            plotted_label_neighbor = False
            for n_info in neighbors_info:
                features = n_info['features']
                label_neighbor = "Najbližší sused" if not plotted_label_neighbor else None
                ax.scatter(features[0], features[1], edgecolor='lime', facecolors='none',
                           s=180, linewidths=2, label=label_neighbor, zorder=4)  # zorder aby boli nad ostatnými
                if label_neighbor: plotted_label_neighbor = True

            ax.set_xlabel("Príznak 1");
            ax.set_ylabel("Príznak 2")

        elif num_features == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            unique_classes = sorted(list(set(item[0][1] for item in all_training_data_with_ids)))
            try:
                cmap = plt.colormaps.get_cmap('viridis')
            except AttributeError:
                cmap = plt.cm.get_cmap('viridis')
            colors = cmap([i / len(unique_classes) for i in range(len(unique_classes))]) if unique_classes else ['blue']
            class_color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}

            plotted_labels_train = set()
            for (features, class_label), original_id in all_training_data_with_ids:
                is_being_classified = (original_id == point_being_classified_original_id)
                marker = 'o'
                size = 70 if is_being_classified else 50
                edge_color_train = 'black' if is_being_classified else None

                label_train = f"Trieda {class_label}" if class_label not in plotted_labels_train else None
                ax.scatter(features[0], features[1], features[2],
                           color=class_color_map.get(class_label, 'gray'),
                           label=label_train, alpha=0.7, s=size, marker=marker, edgecolor=edge_color_train)
                if label_train: plotted_labels_train.add(class_label)

            ax.scatter(point_being_classified_features[0], point_being_classified_features[1],
                       point_being_classified_features[2],
                       color='red', marker='x', s=120, label="Klasifikovaný bod", zorder=5)

            plotted_label_neighbor = False
            for n_info in neighbors_info:
                features = n_info['features']
                label_neighbor = "Najbližší sused" if not plotted_label_neighbor else None
                ax.scatter(features[0], features[1], features[2], edgecolor='lime',
                           facecolors='none', s=180, linewidths=2, label=label_neighbor, zorder=4)
                if label_neighbor: plotted_label_neighbor = True

            ax.set_xlabel("Príznak 1");
            ax.set_ylabel("Príznak 2");
            ax.set_zlabel("Príznak 3")
        else:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Graf je dostupný len pre 2D alebo 3D dáta.",
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xticks([]);
            ax.set_yticks([])

        ax.set_title(ax_title)
        if num_features == 2 or num_features == 3:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))  # Odstráni duplikáty z legendy
            ax.legend(by_label.values(), by_label.keys(), loc='best')

        ax.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()

    def classify(self):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self._plot_initial_message("Spracovávam dáta...")

        try:
            # 1. Parsovanie všetkých tréningových dát
            train_data_str = self.train_data_text.get(1.0, tk.END)
            # parsed_data_with_ids obsahuje ((features, class_label), original_id)
            all_training_data_with_ids, num_features, _ = self._parse_training_data(train_data_str)

            # Prevod na formát, ktorý očakáva calculate_dimension_weights: [(features, class_label), ...]
            full_training_data_for_weights = [item[0] for item in all_training_data_with_ids]

            k_str = self.k_entry.get()
            if not k_str.isdigit() or int(k_str) <= 0: raise ValueError("k musí byť kladné.")
            k = int(k_str)
            distance_metric = self.distance_metric_var.get()

            # 2. Výpočet váh dimenzií VŽDY z plnej tréningovej sady
            current_dim_weights = calculate_dimension_weights(full_training_data_for_weights)
            if not current_dim_weights or len(current_dim_weights) != num_features:
                raise ValueError(f"Váhy dimenzií ({len(current_dim_weights)}) vs príznaky ({num_features}).")

            mode = self.classification_mode.get()
            point_to_classify_features = None
            original_class_of_loo_point = None
            point_being_classified_original_id = None  # Pre graf

            train_data_for_knn_neighbors = []  # Dáta, z ktorých sa budú hľadať susedia

            if mode == "Nový bod":
                new_point_str = self.new_point_entry.get()
                point_to_classify_features = self._parse_new_point(new_point_str, num_features)
                # Susedia sa hľadajú zo všetkých tréningových dát
                train_data_for_knn_neighbors = full_training_data_for_weights
                self.result_text.insert(tk.END, f"--- Klasifikácia nového bodu ---\n")
                self.result_text.insert(tk.END, f"Nový bod: {point_to_classify_features}\n")

            elif mode == "LOO":
                selected_loo_str = self.loo_point_var.get()
                if not selected_loo_str:
                    raise ValueError("Nebol vybraný žiadny bod pre LOO klasifikáciu.")

                # Nájdeme vybraný bod v all_training_data_with_ids
                selected_loo_index = -1
                for i, ((_features, _class_label), _original_id_text) in enumerate(all_training_data_with_ids):
                    # _original_id_text je napr. "X1: (0.0,0.0,0.0) -> T1"
                    # selected_loo_str je to isté
                    if f"{_original_id_text.split(':')[0]}: {_features} -> {_class_label}" == selected_loo_str:
                        selected_loo_index = i
                        break

                if selected_loo_index == -1:
                    raise ValueError(f"Vybraný LOO bod '{selected_loo_str}' sa nenašiel v dátach.")

                (features_loo, class_loo), id_loo = all_training_data_with_ids[selected_loo_index]
                point_to_classify_features = features_loo
                original_class_of_loo_point = class_loo
                point_being_classified_original_id = id_loo

                # Vytvorenie dočasnej sady pre hľadanie susedov (bez LOO bodu)
                train_data_for_knn_neighbors = [
                    item[0] for i, item in enumerate(all_training_data_with_ids) if i != selected_loo_index
                ]
                if not train_data_for_knn_neighbors and k > 0:  # Ak po odobratí neostali žiadni susedia
                    messagebox.showwarning("Upozornenie",
                                           "Po odobratí vybraného bodu neostali žiadni ďalší tréningoví susedia. Klasifikácia nie je možná.")
                    self._plot_initial_message(f"LOO pre {id_loo}: Žiadni susedia.")
                    self.result_text.insert(tk.END,
                                            f"--- LOO Klasifikácia pre bod {id_loo} {features_loo} (orig. trieda: {class_loo}) ---\n")
                    self.result_text.insert(tk.END, "Žiadni susedia pre klasifikáciu.\n")
                    return

                self.result_text.insert(tk.END,
                                        f"--- LOO Klasifikácia pre bod {id_loo} {features_loo} (orig. trieda: {class_loo}) ---\n")

            # 3. Spustenie KNN
            predicted_class, neighbors_info, class_total_weights = get_weighted_knn_prediction(
                train_data_for_knn_neighbors, point_to_classify_features, k, distance_metric, current_dim_weights
            )

            # 4. Zobrazenie výsledkov
            self.result_text.insert(tk.END, f"Metrika: {distance_metric}, k: {k}\n")
            self.result_text.insert(tk.END, f"Váhy dimenzií (W_ed): {[f'{w:.6f}' for w in current_dim_weights]}\n\n")

            if mode == "LOO":
                self.result_text.insert(tk.END, f"Pôvodná trieda bodu: {original_class_of_loo_point}\n")
            self.result_text.insert(tk.END, f"Predikovaná trieda: {predicted_class}\n\n")

            self.result_text.insert(tk.END, "Celkové váhy tried:\n")
            for cl, weight_val in class_total_weights.items():
                w_str = "nekonečno" if weight_val == float('inf') else f"{weight_val:.4f}"
                self.result_text.insert(tk.END, f"  Trieda {cl}: {w_str}\n")

            if neighbors_info:
                self.result_text.insert(tk.END, "\nDetail susedov (z dočasnej sady pre LOO):\n")
                for n in neighbors_info:
                    w_str = "nekonečno" if n['weight'] == float('inf') else f"{n['weight']:.4f}"
                    self.result_text.insert(tk.END,
                                            f"  {n['id']}: {n['features']}, Tr: {n['class']}, "  # n['id'] je tu len dočasné ID suseda
                                            f"Vzd: {n['distance']:.6f}, Váha: {w_str}\n"
                                            )
            else:
                self.result_text.insert(tk.END, "\nŽiadni susedia neboli nájdení (alebo k=0).\n")

            if MATPLOTLIB_AVAILABLE:
                # Pre graf používame všetky pôvodné dáta, aby sa zobrazili všetky body
                self._update_plot(all_training_data_with_ids, point_to_classify_features,
                                  point_being_classified_original_id, neighbors_info, num_features)

        except ValueError as e:
            messagebox.showerror("Chyba vstupu", str(e))
            self.result_text.insert(tk.END, f"CHYBA: {str(e)}\n")
            self._plot_initial_message(f"Chyba vstupu:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Neočekávaná chyba", str(e))
            self.result_text.insert(tk.END, f"NEOČAKÁVANÁ CHYBA: {str(e)}\n")
            self._plot_initial_message(f"Neočekávaná chyba:\n{str(e)}")
        finally:
            self.result_text.config(state=tk.DISABLED)


if __name__ == '__main__':
    root = tk.Tk()
    app_gui = KNN_GUI(root)
    if not MATPLOTLIB_AVAILABLE:
        print("Upozornenie: Matplotlib nie je nainštalovaný. Graf nebude dostupný.")
        print("Inštalácia: pip install matplotlib")
    root.mainloop()
