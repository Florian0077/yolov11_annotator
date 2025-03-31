import cv2
import os
from ultralytics import YOLO
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import uuid
import shutil
import json


def generate_color_palette(num_classes):
    """Génère une palette de couleurs distinctes pour chaque classe."""
    colors = []
    for i in range(num_classes):
        hue = int(180 * i / num_classes)
        color = cv2.cvtColor(
            np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color)))
    return colors


def resize_image(image, max_width, max_height):
    """Redimensionne l'image pour qu'elle tienne dans les dimensions spécifiées."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return resized_image, scale


def create_xml(image_path, selected_predictions, class_names, output_dir):
    """
    Crée un fichier XML correspondant aux prédictions sélectionnées et déplace l'image avec un UUID commun.
    """
    # Charger les dimensions de l'image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Nom de l'image
    image_name = os.path.basename(image_path)
    image_extension = os.path.splitext(image_name)[1]

    # Générer un UUID
    unique_id = str(uuid.uuid4())
    new_image_name = unique_id + image_extension
    new_xml_name = unique_id + '.xml'

    # Déplacer et renommer l'image dans le dossier de sortie
    new_image_path = os.path.join(output_dir, new_image_name)
    shutil.move(image_path, new_image_path)

    # Racine XML
    annotation = Element("annotation")
    SubElement(annotation, "folder").text = "images"
    SubElement(annotation, "filename").text = new_image_name

    # Taille
    size = SubElement(annotation, "size")
    SubElement(size, "width").text = str(width)
    SubElement(size, "height").text = str(height)
    SubElement(size, "depth").text = "3"

    # Objet pour chaque prédiction sélectionnée
    for pred in selected_predictions:
        class_name = pred['class_name']
        x_center, y_center, bbox_width, bbox_height = pred['bbox']

        obj = SubElement(annotation, "object")
        SubElement(obj, "name").text = class_name
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

    # Sauvegarder dans un fichier XML avec le nouveau nom
    xml_str = tostring(annotation)
    parsed = parseString(xml_str)
    output_path = os.path.join(output_dir, new_xml_name)
    with open(output_path, "w") as xml_file:
        xml_file.write(parsed.toprettyxml(indent="  "))
    print(f"XML sauvegardé : {output_path}")


class PredictionApp:
    def __init__(self, root, model, class_names, color_palette, output_dir, confidence_threshold):
        self.root = root
        self.root.title("AutoPredict - Simplifiez la labélisation")
        self.model = model
        self.class_names = class_names
        self.color_palette = color_palette
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.selected_classes = list(range(len(class_names)))  # Toutes les classes sélectionnées par défaut
        self.image_paths = []
        self.current_image_index = 0
        self.predictions = []
        self.check_vars = []
        self.class_vars = []
        self.image = None
        self.photo = None

        # Gérer la fermeture de la fenêtre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Obtenir les dimensions de l'écran
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.max_width = self.screen_width - 100  # Réserver de la place pour les contrôles
        self.max_height = self.screen_height - 300  # Tenir compte de la barre de titre, etc.

        # Appliquer le thème 'clam' pour un look moderne
        style = ttk.Style()
        style.theme_use('clam')

        # Configurer les styles
        style.configure('TLabel', font=('Helvetica', 12))
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('TButton', font=('Helvetica', 12))
        style.configure('Small.TLabel', font=('Helvetica', 10))

        # Créer le cadre principal
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # En-tête avec le logo
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(side=tk.TOP, fill=tk.X)

        # Ajouter le logo
        logo_path = 'logo.png'  # Remplacez par le chemin réel de votre logo
        if os.path.exists(logo_path):
            logo_image = Image.open(logo_path)
            logo_photo = ImageTk.PhotoImage(logo_image)
            self.logo_label = ttk.Label(self.header_frame, image=logo_photo)
            self.logo_label.image = logo_photo  # Conserver une référence
            self.logo_label.pack(side=tk.LEFT, padx=10, pady=10)
        else:
            self.logo_label = ttk.Label(self.header_frame, text="Votre Logo", style='Title.TLabel')
            self.logo_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Cadre du contenu pour l'image et les contrôles
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # Configurer la grille pour le content_frame
        self.content_frame.columnconfigure(0, weight=7)
        self.content_frame.columnconfigure(1, weight=3)
        self.content_frame.rowconfigure(0, weight=1)

        # Canvas pour afficher l'image
        self.canvas_frame = ttk.Frame(self.content_frame)
        self.canvas_frame.grid(row=0, column=0, sticky='nsew')
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, bg='gray')
        self.canvas.grid(row=0, column=0, sticky='nsew')

        # Cadre pour les contrôles
        self.control_frame = ttk.Frame(self.content_frame)
        self.control_frame.grid(row=0, column=1, sticky='ns')

        # Cadre pour les checkboxes avec barre de défilement
        self.checkbox_container_frame = ttk.Frame(self.control_frame)
        self.checkbox_container_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas pour contenir les Checkbuttons et Comboboxes
        self.checkbox_canvas = tk.Canvas(self.checkbox_container_frame)
        self.checkbox_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.checkbox_container_frame, command=self.checkbox_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.checkbox_canvas.configure(yscrollcommand=scrollbar.set)
        self.checkbox_canvas.bind('<Configure>', lambda e: self.checkbox_canvas.configure(scrollregion=self.checkbox_canvas.bbox("all")))

        # Frame à l'intérieur du Canvas pour les widgets
        self.checkbox_frame = ttk.Frame(self.checkbox_canvas)
        self.checkbox_canvas.create_window((0, 0), window=self.checkbox_frame, anchor='nw')

        # Boutons
        self.button_frame = ttk.Frame(self.control_frame)
        self.button_frame.pack(pady=5)

        self.save_button = ttk.Button(self.button_frame, text="Sauvegarder", command=self.save_and_next)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.cancel_button = ttk.Button(self.button_frame, text="Passer", command=self.skip_and_next)
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        # Bouton pour changer de dossier
        self.folder_button = ttk.Button(self.control_frame, text="Changer de dossier", command=self.select_folder)
        self.folder_button.pack(fill=tk.X, pady=5)

        # Slider pour le seuil de détection
        self.slider_frame = ttk.Frame(self.control_frame)
        self.slider_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.slider_frame, text="Seuil de détection:").pack(side=tk.TOP)

        self.confidence_slider = tk.Scale(
            self.slider_frame, from_=0, to=1, orient=tk.HORIZONTAL, resolution=0.001)
        self.confidence_slider.set(self.confidence_threshold)
        self.confidence_slider.pack(fill=tk.X)

        self.confidence_slider.bind("<ButtonRelease-1>", self.update_confidence_threshold)

        # Bouton pour la sélection des classes
        self.class_button = ttk.Button(self.control_frame, text="Sélection des classes", command=self.open_class_selection)
        self.class_button.pack(fill=tk.X, pady=5)

        # Pied de page avec le copyright
        self.footer_frame = ttk.Frame(self.main_frame)
        self.footer_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_label = ttk.Label(self.footer_frame, text="", style='Small.TLabel')
        self.progress_label.pack(side=tk.LEFT, padx=10)

        self.status_label = ttk.Label(self.footer_frame, text="", style='Small.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Remplacez 'Votre Société' par le nom de votre entreprise ou votre nom
        self.company_name = 'Florian0077'
        self.year = '2024'
        copyright_text = f"© {self.year} {self.company_name}"
        copyright_padding = (10, 5)

        # Ajuster l'alignement du copyright
        self.footer_right_frame = ttk.Frame(self.footer_frame)
        self.footer_right_frame.pack(side=tk.RIGHT)

        # Espaceur pour pousser le copyright à droite
        ttk.Label(self.footer_right_frame).pack(side=tk.LEFT, expand=True)

        # Label du copyright
        self.copyright_label = ttk.Label(
            self.footer_right_frame, text=copyright_text, style='Small.TLabel')
        self.copyright_label.pack(
            side=tk.RIGHT, padx=copyright_padding)

        # Sélectionner le dossier initial
        self.select_folder()

    def select_folder(self):
        IMAGES_DIR = filedialog.askdirectory(title="Sélectionnez le dossier contenant les images à traiter")
        if not IMAGES_DIR:
            print("Aucun dossier sélectionné.")
            return

        # Obtenir la liste des images
        self.image_paths = [os.path.join(IMAGES_DIR, img) for img in os.listdir(IMAGES_DIR)
                            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

        if not self.image_paths:
            print("Aucune image trouvée dans le dossier sélectionné.")
            return

        self.current_image_index = 0
        self.load_image()

    def open_class_selection(self):
        # Créer une fenêtre pour la sélection des classes
        class_selection_window = tk.Toplevel(self.root)
        class_selection_window.title("Sélection des classes à détecter")

        # Variables pour stocker l'état de chaque case à cocher
        class_vars = []
        for i, class_name in enumerate(self.class_names):
            var = tk.BooleanVar(value=(i in self.selected_classes))
            cb = ttk.Checkbutton(class_selection_window, text=class_name, variable=var)
            cb.pack(anchor=tk.W)
            class_vars.append((i, var))

        # Bouton OK pour confirmer la sélection
        def confirm_selection():
            self.selected_classes = [i for i, var in class_vars if var.get()]
            if not self.selected_classes:
                print("Aucune classe sélectionnée.")
                class_selection_window.destroy()
                return
            class_selection_window.destroy()
            # Mettre à jour les prédictions avec les nouvelles classes sélectionnées
            self.predictions = []
            self.run_predictions()
            self.update_controls()
            self.display_predictions()

        ok_button = ttk.Button(class_selection_window, text="OK", command=confirm_selection)
        ok_button.pack(pady=10)

    def load_image(self):
        if self.current_image_index >= len(self.image_paths):
            print("Toutes les images ont été traitées.")
            self.root.destroy()
            return

        # Mettre à jour le label de progression
        self.progress_label.config(
            text=f"Image {self.current_image_index + 1} sur {len(self.image_paths)}")

        image_path = self.image_paths[self.current_image_index]

        # Lire l'image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Impossible de lire l'image : {image_path}")
            self.current_image_index += 1
            self.load_image()
            return

        self.image_path = image_path
        self.original_image = image.copy()
        self.image, self.scale = resize_image(
            self.original_image, self.max_width, self.max_height)

        # Réinitialiser les variables
        self.canvas.delete("all")
        self.check_vars = []
        self.class_vars = []
        self.predictions = []

        # Exécuter les prédictions
        self.run_predictions()
        self.update_controls()
        self.display_predictions()

        # Incrémenter l'index de l'image
        self.current_image_index += 1

    def run_predictions(self):
        if not self.selected_classes:
            self.selected_classes = list(range(len(self.class_names)))

        # Prédictions YOLO avec seuil de confiance personnalisé et classes sélectionnées
        results = self.model.predict(
            self.original_image, conf=self.confidence_threshold, classes=self.selected_classes)
        predictions = results[0].boxes.xywhn.cpu().numpy()  # Coordonnées normalisées
        class_ids = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        # Préparer les prédictions
        for class_id, bbox_norm, conf in zip(class_ids, predictions, confidences):
            x_center, y_center, bbox_width, bbox_height = bbox_norm
            x1 = int((x_center - bbox_width / 2) * self.original_image.shape[1])
            y1 = int((y_center - bbox_height / 2) * self.original_image.shape[0])
            x2 = int((x_center + bbox_width / 2) * self.original_image.shape[1])
            y2 = int((y_center + bbox_height / 2) * self.original_image.shape[0])
            self.predictions.append({
                'class_id': int(class_id),
                'bbox': bbox_norm,
                'bbox_coords': (x1, y1, x2, y2),
                'conf': conf
            })

        # Ajuster les coordonnées des boîtes en fonction de l'échelle
        for pred in self.predictions:
            x1, y1, x2, y2 = pred['bbox_coords']
            pred['bbox_coords'] = (
                int(x1 * self.scale),
                int(y1 * self.scale),
                int(x2 * self.scale),
                int(y2 * self.scale)
            )

    def update_controls(self):
        # Effacer les widgets précédents
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()
        self.check_vars = []
        self.class_vars = []

        # Créer les contrôles
        self.create_controls()

    def display_predictions(self):
        # Effacer le canvas
        self.canvas.delete("all")

        # Convertir l'image pour Tkinter
        self.photo = ImageTk.PhotoImage(
            image=Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)))

        # Ajuster la taille du canvas à l'image
        self.canvas.config(width=self.photo.width(), height=self.photo.height())

        # Afficher l'image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Dessiner les boîtes sur le canvas
        self.draw_boxes()

    def create_controls(self):
        # Tri des prédictions selon leur position dans l'image (haut à bas, gauche à droite)
        self.predictions.sort(key=lambda pred: (pred['bbox_coords'][1], pred['bbox_coords'][0]))

        for idx, pred in enumerate(self.predictions):
            # Variable pour la case à cocher
            var = tk.BooleanVar(value=True)
            self.check_vars.append(var)

            # Variable pour la combobox
            class_var = tk.StringVar(value=self.class_names[int(pred['class_id'])])
            self.class_vars.append(class_var)

            # Frame pour contenir la case à cocher et la combobox
            pred_frame = ttk.Frame(self.checkbox_frame)
            pred_frame.pack(anchor=tk.W, pady=2)

            # Case à cocher
            cb = ttk.Checkbutton(pred_frame, variable=var, command=self.update_selection)
            cb.pack(side=tk.LEFT)

            # Combobox pour sélectionner la classe
            combo = ttk.Combobox(pred_frame, textvariable=class_var, values=self.class_names, state='readonly')
            combo.pack(side=tk.LEFT, padx=5)
            combo.bind('<<ComboboxSelected>>', lambda event, idx=idx: self.update_class(idx))

    def draw_boxes(self):
        for idx, pred in enumerate(self.predictions):
            class_id = pred['class_id']
            class_name = self.class_names[int(class_id)]
            pred['class_name'] = class_name  # Stocker le nom de la classe

            x1, y1, x2, y2 = pred['bbox_coords']
            color_rgb = self.color_palette[int(class_id)]
            color = '#%02x%02x%02x' % color_rgb
            rect_id = self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=color, width=2)

            # Affichage du libellé à gauche du cadre
            text = class_name
            # Mesurer la largeur du texte
            font = ("Arial", 12, "bold")
            temp_text_id = self.canvas.create_text(0, 0, text=text, font=font, anchor=tk.NW)
            bbox = self.canvas.bbox(temp_text_id)
            self.canvas.delete(temp_text_id)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_x = x1 - text_width - 10  # 10 pixels à gauche du rectangle
            text_y = y1 + (y2 - y1) / 2 - text_height / 2  # Centré verticalement

            # Vérifier si le label sort de la zone visible sur la gauche
            if text_x < 0:
                # Placer le label à droite du rectangle
                text_x = x2 + 10  # 10 pixels à droite du rectangle

            # Vérifier que le label ne dépasse pas le bord droit de l'image
            if text_x + text_width > self.canvas.winfo_width():
                text_x = self.canvas.winfo_width() - text_width - 5  # Ajuster pour rester dans l'image

            text_id = self.canvas.create_text(
                text_x, text_y, anchor=tk.NW, text=text, fill=color, font=font)

            # Stocker les identifiants pour pouvoir les manipuler
            pred['rect_id'] = rect_id
            pred['text_id'] = text_id

    def update_selection(self):
        # Mettre à jour la visibilité des rectangles en fonction des cases cochées
        for var, pred in zip(self.check_vars, self.predictions):
            state = 'normal' if var.get() else 'hidden'
            self.canvas.itemconfigure(pred['rect_id'], state=state)
            self.canvas.itemconfigure(pred['text_id'], state=state)

    def update_class(self, idx):
        # Obtenir le nouveau nom de classe depuis la combobox
        new_class_name = self.class_vars[idx].get()
        # Trouver l'index du nouveau nom de classe dans la liste des classes
        new_class_id = self.class_names.index(new_class_name)

        # Mettre à jour les données de la prédiction
        pred = self.predictions[idx]
        pred['class_id'] = new_class_id
        pred['class_name'] = new_class_name

        # Mettre à jour la couleur
        color_rgb = self.color_palette[new_class_id]
        color = '#%02x%02x%02x' % color_rgb

        # Mettre à jour la couleur du rectangle
        self.canvas.itemconfigure(pred['rect_id'], outline=color)

        # Supprimer l'ancien texte
        self.canvas.delete(pred['text_id'])

        # Recalculer la position du texte
        x1, y1, x2, y2 = pred['bbox_coords']
        text = new_class_name
        font = ("Arial", 12, "bold")
        temp_text_id = self.canvas.create_text(0, 0, text=text, font=font, anchor=tk.NW)
        bbox = self.canvas.bbox(temp_text_id)
        self.canvas.delete(temp_text_id)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_x = x1 - text_width - 10  # 10 pixels à gauche du rectangle
        text_y = y1 + (y2 - y1) / 2 - text_height / 2  # Centré verticalement

        # Vérifier si le label sort de la zone visible sur la gauche
        if text_x < 0:
            # Placer le label à droite du rectangle
            text_x = x2 + 10  # 10 pixels à droite du rectangle

        # Vérifier que le label ne dépasse pas le bord droit de l'image
        if text_x + text_width > self.canvas.winfo_width():
            text_x = self.canvas.winfo_width() - text_width - 5  # Ajuster pour rester dans l'image

        # Redessiner le texte avec la nouvelle classe et la nouvelle couleur
        text_id = self.canvas.create_text(
            text_x, text_y, anchor=tk.NW, text=text, fill=color, font=font)
        pred['text_id'] = text_id

    def save_and_next(self):
        # Filtrer les prédictions sélectionnées
        selected_preds = [
            pred for pred, var in zip(self.predictions, self.check_vars) if var.get()]
        if selected_preds:
            create_xml(self.image_path, selected_preds,
                       self.class_names, self.output_dir)
            self.status_label.config(text="Sauvegardé avec succès", foreground='green')
        else:
            print("Aucune prédiction sélectionnée. Aucun fichier XML créé.")
            self.status_label.config(text="Aucune prédiction sélectionnée", foreground='orange')
        # Charger l'image suivante
        self.load_image()

    def skip_and_next(self):
        # Ignorer l'image actuelle et charger la suivante
        print("Image ignorée.")
        self.status_label.config(text="Image ignorée", foreground='blue')
        self.load_image()

    def update_confidence_threshold(self, event=None):
        # Mettre à jour le seuil de confiance et recharger les prédictions
        self.confidence_threshold = self.confidence_slider.get()
        self.status_label.config(
            text=f"Seuil de confiance: {self.confidence_threshold:.2f}", foreground='black')
        self.predictions = []
        self.run_predictions()
        self.update_controls()
        self.display_predictions()

    def on_closing(self):
        # Arrêter l'application
        self.root.destroy()


def main(model_path, output_dir, class_names, confidence_threshold=0.25):
    # Créer l'interface Tkinter
    root = tk.Tk()

    # Charger le modèle YOLO
    model = YOLO(model_path)

    # Générer une palette de couleurs
    color_palette = generate_color_palette(len(class_names))

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Créer et lancer l'interface Tkinter
    app = PredictionApp(root, model, class_names,
                        color_palette, output_dir, confidence_threshold)
    root.mainloop()


import json
import os

if __name__ == "__main__":
    # Charger la configuration depuis un fichier JSON
    config_path = os.path.join(os.getcwd(), 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extraction des variables de configuration
    PATH = os.getcwd()
    MODEL_PATH = os.path.join(PATH, config.get('model_dir', 'model'), config.get('model_file', 'best.pt'))
    OUTPUT_DIR = os.path.join(PATH, config.get('output_dir', 'auto_dataset'))
    CLASS_NAMES = config.get('class_names', [])
    
    # Récupération du seuil de confiance depuis la configuration ou utilisation de la valeur par défaut
    confidence_threshold = config.get('confidence_threshold', 0.2)
    
    # Appel de la fonction principale
    main(MODEL_PATH, OUTPUT_DIR, CLASS_NAMES, confidence_threshold=confidence_threshold)
