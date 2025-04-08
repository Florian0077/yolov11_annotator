import cv2
import os
from ultralytics import YOLO
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString


def resize_image(image, max_width=800, max_height=800):
    """
    Redimensionne une image pour qu'elle s'adapte à une taille maximale tout en conservant les proportions.
    """
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    if scale < 1:  # Redimensionne uniquement si nécessaire
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        return resized_image
    return image


def create_xml(image_path, predictions, class_names, output_dir):
    """
    Crée un fichier XML correspondant aux prédictions YOLO.
    """
    # Charger les dimensions de l'image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Nom de l'image
    image_name = os.path.basename(image_path)

    # Racine XML
    annotation = Element("annotation")
    SubElement(annotation, "folder").text = "images"
    SubElement(annotation, "filename").text = image_name

    # Taille
    size = SubElement(annotation, "size")
    SubElement(size, "width").text = str(width)
    SubElement(size, "height").text = str(height)
    SubElement(size, "depth").text = "3"

    # Objet pour chaque prédiction
    for class_id, (x_center, y_center, bbox_width, bbox_height) in predictions:
        obj = SubElement(annotation, "object")
        SubElement(obj, "name").text = class_names[int(class_id)]
        SubElement(obj, "pose").text = "Unspecified"
        SubElement(obj, "truncated").text = "0"
        SubElement(obj, "difficult").text = "0"

        # Convertir les coordonnées YOLO en coordonnées XML
        xmin = int((x_center - bbox_width / 2) * width)
        ymin = int((y_center - bbox_height / 2) * height)
        xmax = int((x_center + bbox_width / 2) * width)
        ymax = int((y_center + bbox_height / 2) * height)

        bndbox = SubElement(obj, "bndbox")
        SubElement(bndbox, "xmin").text = str(xmin)
        SubElement(bndbox, "ymin").text = str(ymin)
        SubElement(bndbox, "xmax").text = str(xmax)
        SubElement(bndbox, "ymax").text = str(ymax)

    # Sauvegarder dans un fichier XML
    xml_str = tostring(annotation)
    parsed = parseString(xml_str)
    output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.xml")
    with open(output_path, "w") as xml_file:
        xml_file.write(parsed.toprettyxml(indent="  "))
    print(f"XML sauvegardé : {output_path}")


def main(model_path, images_dir, output_dir, class_names):
    # Charger le modèle YOLO
    model = YOLO(model_path)

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Traiter chaque image
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)

        # Lire et afficher l'image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Impossible de lire l'image : {image_path}")
            continue

        # Prédictions YOLO
        results = model.predict(image, conf=0.25)
        predictions = results[0].boxes.xywhn.cpu().numpy()  # Coordonnées normalisées
        class_ids = results[0].boxes.cls.cpu().numpy()

        # Afficher les prédictions sur l'image
        for class_id, (x_center, y_center, bbox_width, bbox_height) in zip(
            class_ids, predictions
        ):
            if 0 <= int(class_id) < len(class_names):
                label = f"{class_names[int(class_id)]}"
            else:
                label = f"Unknown ({int(class_id)})"
                print(f"Classe prédite inconnue: {class_id}")

            # Dessiner les rectangles sur l'image
            x1 = int((x_center - bbox_width / 2) * image.shape[1])
            y1 = int((y_center - bbox_height / 2) * image.shape[0])
            x2 = int((x_center + bbox_width / 2) * image.shape[1])
            y2 = int((y_center + bbox_height / 2) * image.shape[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Redimensionner l'image pour l'affichage
        resized_image = resize_image(image, max_width=800, max_height=800)

        # Afficher l'image redimensionnée
        cv2.imshow("Predictions", resized_image)
        key = cv2.waitKey(0)  # Attendre une touche

        # Si validé par l'utilisateur, créer le fichier XML
        if key == ord("y"):  # Appuyer sur 'y' pour valider
            print("Validation acceptée. Création du fichier XML...")
            create_xml(image_path, zip(class_ids, predictions), class_names, output_dir)
        elif key == ord("n"):  # Appuyer sur 'n' pour refuser
            print("Validation refusée. Aucun fichier XML créé.")
        elif key == ord("q"):  # Appuyer sur 'q' pour quitter
            print("Sortie du programme.")
            break

    # Fermer toutes les fenêtres
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = (
        r"F:yoloDoc\runs\detect\train\weights\best.pt"  # Chemin vers le modèle entraîné
    )
    IMAGES_DIR = (
        r"F:yoloDoc\output\page_001"  # Répertoire contenant les nouvelles images
    )
    OUTPUT_DIR = (
        r"F:yoloDoc\auto_dataset\xml"  # Répertoire de sortie pour les fichiers XML
    )
    CLASS_NAMES = [
        "all",
        "ecm_data",
        "header",
        "context_table",
        "ecm_history",
        "context_adapt",
        "footer",
        "ecm_validation",
    ]  # Remplacez par vos classes

    main(MODEL_PATH, IMAGES_DIR, OUTPUT_DIR, CLASS_NAMES)
