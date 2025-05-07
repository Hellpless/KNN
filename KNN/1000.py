import random


def generate_knn_test_data(num_rows=1000):
    """
    Generuje testovacie dáta pre KNN klasifikátor.
    Každý riadok má 3 číselné príznaky a jednu z troch tried (T1, T2, T3).
    """
    data_lines = []
    class_labels = ["T1", "T2", "T3"]

    for i in range(num_rows):
        # Generovanie náhodných príznakov
        # Môžeš si upraviť rozsahy podľa potreby
        feature1 = round(random.uniform(0, 10), 2)  # Napr. od 0 do 10
        feature2 = round(random.uniform(0, 100), 2)  # Napr. od 0 do 100
        feature3 = round(random.uniform(0, 50), 2)  # Napr. od 0 do 50

        # Náhodný výber triedy
        class_label = random.choice(class_labels)

        data_lines.append(f"{feature1},{feature2},{feature3},{class_label}")

    return "\n".join(data_lines)


if __name__ == '__main__':
    # Vygenerovanie 1000 riadkov dát
    generated_data = generate_knn_test_data(1000)

    # Výpis dát (v reálnej aplikácii by si ich uložil do súboru alebo použil priamo)
    # print(generated_data)

    # Pre ukážku ich uložím do súboru, aby si ich mohol ľahko skopírovať
    try:
        with open("knn_dummy_data_1000.txt", "w") as f:
            f.write(generated_data)
        print("Dáta boli úspešne vygenerované do súboru 'knn_dummy_data_1000.txt'")
    except IOError:
        print("Chyba pri zápise dát do súboru.")
        print("\nTu sú dáta na skopírovanie:\n")
        print(generated_data)

