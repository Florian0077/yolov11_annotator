from pathlib import Path
from pdf2image import convert_from_path
import os


def convert_pdf_to_png(pdf_dir, output_dir, dpi=300):
    """
    Convertit tous les fichiers PDF d'un dossier en images PNG,
    organisées par numéro de page.

    Args:
        pdf_dir (str): Chemin du dossier contenant les PDF
        output_dir (str): Chemin du dossier de sortie pour les PNG
        dpi (int): Résolution des images en DPI (défaut: 300)
    """
    # Dictionnaire pour suivre le nombre maximum de pages
    max_pages = 0

    # Première passe : déterminer le nombre maximum de pages
    for pdf_file in Path(pdf_dir).glob("*.pdf"):
        try:
            pages = convert_from_path(pdf_file, dpi=1)  # DPI bas pour rapidité
            max_pages = max(max_pages, len(pages))
        except Exception as e:
            print(f"Erreur lors de l'analyse de {pdf_file.name}: {str(e)}")

    # Créer les dossiers pour chaque numéro de page
    for page_num in range(1, max_pages + 1):
        page_dir = os.path.join(output_dir, f"page_{page_num:03d}")
        os.makedirs(page_dir, exist_ok=True)

    # Traiter chaque PDF
    for pdf_file in Path(pdf_dir).glob("*.pdf"):
        print(f"Traitement de {pdf_file.name}...")

        try:
            # Convertir le PDF en images
            pages = convert_from_path(pdf_file, dpi=dpi, fmt="png")

            # Sauvegarder chaque page dans le dossier correspondant
            for i, page in enumerate(pages, 1):
                # Créer le nom du fichier de sortie
                page_dir = os.path.join(output_dir, f"page_{i:03d}")
                output_file = os.path.join(page_dir, f"{pdf_file.stem}.png")

                # Sauvegarder l'image
                page.save(output_file, "PNG")
                print(f"  Page {i} de {pdf_file.name} sauvegardée dans {page_dir}")

        except Exception as e:
            print(f"Erreur lors du traitement de {pdf_file.name}: {str(e)}")
            continue

        print(f"Conversion terminée pour {pdf_file.name}")

    print(
        f"""
Traitement terminé pour tous les fichiers PDF
Nombre total de dossiers créés : {max_pages}
Structure des dossiers :
{output_dir}/
  ├── page_001/
  │   ├── document1.png
  │   ├── document2.png
  │   └── ...
  ├── page_002/
  │   ├── document1.png
  │   ├── document2.png
  │   └── ...
  └── ...
"""
    )


def analyze_result(output_dir):
    """
    Analyse et affiche un résumé de la conversion
    """
    total_files = 0
    page_counts = {}

    for page_dir in Path(output_dir).glob("page_*"):
        count = len(list(page_dir.glob("*.png")))
        page_counts[page_dir.name] = count
        total_files += count

    print("\nRésumé de la conversion :")
    print(f"Nombre total d'images : {total_files}")
    for page, count in sorted(page_counts.items()):
        print(f"{page}: {count} documents")


if __name__ == "__main__":
    # Configuration des chemins
    PDF_DIR = "input"  # Dossier contenant les PDF
    OUTPUT_DIR = "output"  # Dossier de sortie pour les PNG

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Lancer la conversion
    convert_pdf_to_png(PDF_DIR, OUTPUT_DIR)

    # Analyser les résultats
    analyze_result(OUTPUT_DIR)
