import os
import xml.etree.ElementTree as ET
import pandas as pd


def list_classes_in_dataset(folder_path):
    """
    Parcourt tous les fichiers XML dans un dossier et liste les classes présentes et leur occurrence.

    :param folder_path: Chemin vers le dossier contenant les fichiers XML.
    :return: DataFrame des classes et de leur occurrence.
    """
    if not os.path.exists(folder_path):
        print(f"Le dossier {folder_path} n'existe pas.")
        return None

    class_counts = {}
    files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]

    if not files:
        print("Aucun fichier XML trouvé dans le dossier.")
        return None

    for file in files:
        file_path = os.path.join(folder_path, file)

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Parcourir les objets pour extraire les noms de classes
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1

        except ET.ParseError as e:
            print(f"Erreur lors du traitement du fichier {file}: {e}")
        except Exception as e:
            print(f"Une erreur s'est produite avec le fichier {file}: {e}")

    # Créer un DataFrame pour afficher les classes et leurs occurrences
    df = pd.DataFrame(list(class_counts.items()), columns=["Classe", "Occurrence"])
    df = df.sort_values(by="Occurrence", ascending=False).reset_index(drop=True)

    return df


def update_class_in_files(folder_path, old_class, new_class):
    """
    Met à jour les fichiers XML pour remplacer une classe par une autre.

    :param folder_path: Chemin vers le dossier contenant les fichiers XML.
    :param old_class: Nom de la classe à remplacer.
    :param new_class: Nouveau nom de la classe.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]

    for file in files:
        file_path = os.path.join(folder_path, file)

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Remplacer les noms de classe
            updated = False
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name == old_class:
                    obj.find("name").text = new_class
                    updated = True

            if updated:
                # Sauvegarder les modifications
                tree.write(file_path, encoding="utf-8", xml_declaration=True)
                print(f"Classe mise à jour dans le fichier {file}")

        except ET.ParseError as e:
            print(f"Erreur lors du traitement du fichier {file}: {e}")
        except Exception as e:
            print(f"Une erreur s'est produite avec le fichier {file}: {e}")


# Exemple d'utilisation
path = os.getcwd()
folder = "dataset"
folder_path = os.path.join(path, folder)
print(f"Contrôle des éléments de {folder_path}")

# Étape 1 : Lister les classes
df_classes = list_classes_in_dataset(folder_path)
if df_classes is not None:
    import ace_tools_open as tools

    tools.display_dataframe_to_user(
        name="Classes dans les fichiers XML", dataframe=df_classes
    )

# Étape 2 : Prompt pour modifier les classes
if df_classes is not None:
    print("\nVoici les classes disponibles :")
    print(df_classes)

    old_class = input("Entrez le nom de la classe à modifier : ")
    new_class = input(f"Entrez le nouveau nom pour la classe '{old_class}' : ")

    update_class_in_files(folder_path, old_class, new_class)
    print(
        f"\nLa classe '{old_class}' a été remplacée par '{new_class}' dans les fichiers XML."
    )
