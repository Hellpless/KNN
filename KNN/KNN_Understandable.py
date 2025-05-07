import math
from collections import defaultdict

# Krok 1: Príprava dát a výpočet váh dimenzií (Feature Weights)

# Tréningové dáta: ( (feature1, feature2), trieda )
training_data = [
    ((1.77, 80), "M"),
    ((1.76, 49), "F"),
    ((1.71, 56), "F"),
    ((1.85, 96), "M"),
]

# Nový bod na klasifikáciu
new_point_features = (1.68, 58)


def calculate_dimension_weights(data):
    """
    Vypočíta váhy dimenzií (W_ed) podľa vzorca z obrázka:
    W_ed = (priemer štvorcov hodnôt v dimenzii d)^(-1)
    """
    if not data:
        return []
    num_samples = len(data)
    num_dims = len(data[0][0])
    dimension_weights = []
    for d_idx in range(num_dims):
        sum_of_squares = 0
        for sample_features, _ in data:
            sum_of_squares += sample_features[d_idx] ** 2
        mean_of_squares = sum_of_squares / num_samples
        if mean_of_squares == 0:
            print(f"Upozornenie: Priemer štvorcov pre dimenziu {d_idx} je 0. Váha bude float('inf').")
            dimension_weights.append(float('inf'))
        else:
            dimension_weights.append(1 / mean_of_squares)
    return dimension_weights


dim_weights = calculate_dimension_weights(training_data)
print(f"Vypočítané váhy dimenzií (W_ed): {dim_weights}")
print("-" * 30)


# Krok 2: Implementácia funkcií pre výpočet vzdialeností

def weighted_euclidean_distance(point1_features, point2_features, dimension_feature_weights):
    if len(point1_features) != len(point2_features) or len(point1_features) != len(dimension_feature_weights):
        raise ValueError("Počet dimenzií bodov a váh sa musí zhodovať.")
    sum_sq_diff = 0
    for i in range(len(point1_features)):
        diff = point1_features[i] - point2_features[i]
        sum_sq_diff += dimension_feature_weights[i] * (diff ** 2)
    return math.sqrt(sum_sq_diff)


def weighted_manhattan_distance(point1_features, point2_features, dimension_feature_weights):
    if len(point1_features) != len(point2_features) or len(point1_features) != len(dimension_feature_weights):
        raise ValueError("Počet dimenzií bodov a váh sa musí zhodovať.")
    total_distance = 0
    for i in range(len(point1_features)):
        diff = abs(point1_features[i] - point2_features[i])
        total_distance += dimension_feature_weights[i] * diff
    return total_distance


def weighted_chebyshev_distance(point1_features, point2_features, dimension_feature_weights):
    if len(point1_features) != len(point2_features) or len(point1_features) != len(dimension_feature_weights):
        raise ValueError("Počet dimenzií bodov a váh sa musí zhodovať.")
    max_weighted_diff = 0
    for i in range(len(point1_features)):
        diff = abs(point1_features[i] - point2_features[i])
        weighted_diff = dimension_feature_weights[i] * diff
        if weighted_diff > max_weighted_diff:
            max_weighted_diff = weighted_diff
    return max_weighted_diff


# Krok 3: Hlavná logika váhovaného KNN

def get_weighted_knn_prediction(train_data, point_to_classify_features, k, distance_func, current_dim_weights):
    """
    Klasifikuje nový bod pomocou váhovaného KNN.

    Args:
        train_data (list): Tréningové dáta vo formáte [((features), class_label), ...].
        point_to_classify_features (tuple): Príznaky bodu, ktorý sa má klasifikovať.
        k (int): Počet najbližších susedov. Musí byť > 0 a <= počet tréningových dát.
        distance_func (function): Funkcia na výpočet vzdialenosti.
        current_dim_weights (list): Vypočítané váhy dimenzií (W_ed).

    Returns:
        str: Predikovaná trieda.
        list: Zoznam k najbližších susedov s ich vzdialenosťami a váhami.
        dict: Slovník s celkovými váhami pre každú triedu.
    """
    if k <= 0:
        raise ValueError("k musí byť kladné celé číslo.")
    if k > len(train_data):
        # print(f"Upozornenie: k ({k}) je väčšie ako počet tréningových dát ({len(train_data)}). Použije sa k={len(train_data)}.")
        k = len(train_data)

    distances = []
    for i, (train_sample_features, train_sample_class) in enumerate(train_data):
        dist = distance_func(train_sample_features, point_to_classify_features, current_dim_weights)
        # Ukladáme aj pôvodný index pre lepšiu identifikáciu bodu X1, X2...
        distances.append(((train_sample_features, train_sample_class, f"X{i + 1}"), dist))

    distances.sort(key=lambda x: x[1])

    neighbors = distances[:k]

    class_weights = defaultdict(float)
    detailed_neighbors_info = []

    for neighbor_data, dist in neighbors:
        neighbor_features, neighbor_class, neighbor_id = neighbor_data

        weight = 0
        if dist == 0:
            # Ak je vzdialenosť 0, tento sused má "nekonečnú" váhu.
            # Ak je viac takýchto susedov, ich váhy sa sčítajú.
            # Ak majú rôzne triedy, výsledok môže byť nejednoznačný bez ďalšej logiky.
            print(f"Upozornenie: Vzdialenosť k susedovi {neighbor_id} {neighbor_features} ({neighbor_class}) je 0.")
            weight = float('inf')
        else:
            weight = 1 / dist

        class_weights[neighbor_class] += weight
        detailed_neighbors_info.append({
            "id": neighbor_id,
            "features": neighbor_features,
            "class": neighbor_class,
            "distance": dist,
            "weight": weight
        })

    if not class_weights:
        # Toto by sa nemalo stať, ak k > 0 a train_data nie je prázdne
        return "Neznáma (žiadni susedia alebo váhy)", [], {}

    # Nájdenie triedy s najvyššou váhou
    predicted_class = max(class_weights, key=class_weights.get)

    return predicted_class, detailed_neighbors_info, class_weights


# Krok 4: Testovanie a overenie s rôznymi k

print("\n--- Testovanie hlavnej logiky KNN s rôznymi k ---")

# Zoznam hodnôt k, ktoré chceme testovať
k_values_to_test = [1, 2, 3, 4]

# Metriky vzdialenosti na testovanie
distance_metrics = {
    "Váhovaná Euklidovská": weighted_euclidean_distance,
    "Váhovaná Manhattan": weighted_manhattan_distance,
    "Váhovaná Čebyševova": weighted_chebyshev_distance,
}

for metric_name, distance_function in distance_metrics.items():
    print(f"\n===== Testovanie s metrikou: {metric_name} =====")
    for k_val in k_values_to_test:
        if k_val > len(training_data):  # Zabezpečíme, aby k nebolo väčšie ako počet dát
            print(f"\nPreskakujem k={k_val}, pretože je väčšie ako počet tréningových dát ({len(training_data)}).")
            continue

        print(f"\n--- Testovanie pre k = {k_val} ---")

        predicted_class, neighbors_info, cl_weights = get_weighted_knn_prediction(
            training_data,
            new_point_features,
            k_val,
            distance_function,
            dim_weights
        )

        print(f"Predikovaná trieda pre X0={new_point_features}: {predicted_class}")
        print("Celkové váhy tried:", dict(cl_weights))
        print("Detail najbližších susedov:")
        for neighbor in neighbors_info:
            # Formátovanie váhy, aby sa 'inf' pekne zobrazilo
            weight_str = "nekonečno" if neighbor['weight'] == float('inf') else f"{neighbor['weight']:.4f}"
            print(f"  Sused {neighbor['id']}: {neighbor['features']}, Trieda: {neighbor['class']}, "
                  f"Vzdialenosť: {neighbor['distance']:.6f}, Váha: {weight_str}")
    print("=" * 50)

# Poznámka k overeniu s obrázkom pre Euklidovskú vzdialenosť a k=2:
# Na obrázku: Susedia X3(F), X2(F). Váhy: X3=30.96, X2=7.63. Celková váha F=38.59. Predikcia F.
# Náš kód by mal dať rovnaký výsledok.

# Ďalšie kroky:
# - Zvážiť GUI.
# - Testovanie s inými dátami.
# - Prípadne zbalenie do triedy pre lepšiu organizáciu.
