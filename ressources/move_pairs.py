import os
from pathlib import Path
import shutil


def move_paired_files(source_dir, destination_dir, page):
    """
    Déplace les paires de fichiers PNG/XML vers un dossier de destination.
    Ne déplace que les fichiers PNG qui ont un fichier XML correspondant.

    Args:
        source_dir (str): Chemin du dossier source
        destination_dir (str): Chemin du dossier de destination
    """
    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(destination_dir, exist_ok=True)

    # Compter les fichiers pour les statistiques
    stats = {"png_total": 0, "xml_total": 0, "pairs_moved": 0, "png_without_xml": 0}

    # Lister tous les fichiers PNG
    source_path = Path(source_dir)
    png_files = list(source_path.glob("*.png"))

    print(f"Analyse du dossier source : {source_dir}")

    for png_file in png_files:
        stats["png_total"] += 1
        xml_file = png_file.with_suffix(".xml")

        if xml_file.exists():
            stats["xml_total"] += 1
            try:
                # Nouveau nom de fichier avec le numéro de page
                new_png_name = f"{page}_{png_file.name}"
                new_xml_name = f"{page}_{xml_file.name}"

                # Définir les chemins de destination avec les nouveaux noms
                png_dest = Path(destination_dir) / new_png_name
                xml_dest = Path(destination_dir) / new_xml_name

                # Déplacer les deux fichiers avec les nouveaux noms
                shutil.move(str(png_file), str(png_dest))
                shutil.move(str(xml_file), str(xml_dest))

                print(f"Déplacé avec succès: {new_png_name} et {new_xml_name}")
                stats["pairs_moved"] += 1

            except Exception as e:
                print(f"Erreur lors du déplacement de {png_file.name}: {str(e)}")
        else:
            print(f"Pas de fichier XML trouvé pour: {png_file.name}")
            stats["png_without_xml"] += 1

    # Afficher les statistiques
    print("\nRésumé des opérations:")
    print(f"Nombre total de fichiers PNG trouvés: {stats['png_total']}")
    print(f"Nombre de paires PNG/XML trouvées: {stats['pairs_moved']}")
    print(f"Nombre de PNG sans XML correspondant: {stats['png_without_xml']}")

    # Vérifier s'il reste des fichiers XML orphelins
    remaining_xml = list(source_path.glob("*.xml"))
    if remaining_xml:
        print(
            f"\nAttention: {len(remaining_xml)} fichier(s) XML trouvé(s) sans PNG correspondant:"
        )
        for xml_file in remaining_xml:
            print(f"- {xml_file.name}")


def verify_moved_files(destination_dir):
    """
    Vérifie que tous les fichiers dans le dossier de destination sont bien par paires
    """
    dest_path = Path(destination_dir)
    png_files = set(f.stem for f in dest_path.glob("*.png"))
    xml_files = set(f.stem for f in dest_path.glob("*.xml"))

    if png_files == xml_files:
        print(
            "\nVérification réussie: Toutes les paires sont complètes dans le dossier de destination"
        )
    else:
        print(
            "\nAvertissement: Certaines paires pourraient être incomplètes dans le dossier de destination"
        )
        png_only = png_files - xml_files
        xml_only = xml_files - png_files

        if png_only:
            print(f"PNG sans XML: {', '.join(png_only)}")
        if xml_only:
            print(f"XML sans PNG: {', '.join(xml_only)}")


if __name__ == "__main__":


    dossier_sync = ['001', '002', '007', '008', '010', '009', '021']

    for d in dossier_sync :
        # Configuration des chemins
        SOURCE_DIR = f"C:/Users/H14376/Desktop/yoloDoc/output/page_{d}"  # Dossier contenant les fichiers
        DESTINATION_DIR = (
            r"C:\Users\H14376\Desktop\yoloDoc\dataset"  # Dossier de destination
        )

        # Exécuter le déplacement
        move_paired_files(SOURCE_DIR, DESTINATION_DIR, d)

        # Vérifier le résultat
        verify_moved_files(DESTINATION_DIR)
