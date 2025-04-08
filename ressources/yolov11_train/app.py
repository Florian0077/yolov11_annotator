import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import yaml
import random
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from threading import Thread
import json
import pandas as pd
from tkinter import simpledialog


class YOLOTrainingPreparator:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.classes = set()
        self.dataset_dir = self.data_dir / "dataset"

    def create_directory_structure(self):
        """Crée la structure de dossiers nécessaire pour YOLOv11"""
        dirs = {
            "images/train": self.dataset_dir / "images" / "train",
            "images/val": self.dataset_dir / "images" / "val",
            "labels/train": self.dataset_dir / "labels" / "train",
            "labels/val": self.dataset_dir / "labels" / "val",
        }

        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        return dirs

    def convert_coordinates(self, size, box):
        """Convertit les coordonnées du format XML (xmin, ymin, xmax, ymax) au format YOLO (x_center, y_center, width, height)"""
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]

        xmin, ymin, xmax, ymax = box
        w = xmax - xmin
        h = ymax - ymin
        x = xmin + w / 2
        y = ymin + h / 2

        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh

        return x, y, w, h

    def convert_xml_to_yolo(self, xml_file):
        """Convertit un fichier XML en format YOLO"""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Obtenir les dimensions de l'image
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        yolo_lines = []

        # Parcourir tous les objets
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            self.classes.add(class_name)

            # Obtenir l'index de la classe
            class_id = list(sorted(self.classes)).index(class_name)

            # Obtenir les coordonnées
            xmlbox = obj.find("bndbox")
            box = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymax").text),
            )

            # Convertir au format YOLO
            bb = self.convert_coordinates((width, height), box)
            yolo_lines.append(
                f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}"
            )

        return yolo_lines

    def prepare_dataset(self, split_ratio=0.2, progress_callback=None):
        """Prépare le dataset pour l'entraînement"""
        dirs = self.create_directory_structure()

        # Lister tous les fichiers XML
        xml_files = list(self.data_dir.glob("**/*.xml"))
        random.shuffle(xml_files)

        # Calculer le split
        split_index = int(len(xml_files) * (1 - split_ratio))
        train_files = xml_files[:split_index]
        val_files = xml_files[split_index:]

        # Traiter les fichiers d'entraînement
        if progress_callback:
            progress_callback("Préparation des données d'entraînement...", 0)
        else:
            print("Préparation des données d'entraînement...")
            
        total_files = len(train_files) + len(val_files)
        processed = 0
            
        for xml_file in train_files:
            self._process_file(xml_file, "train", dirs)
            processed += 1
            if progress_callback:
                progress_callback(None, processed / total_files * 100)

        # Traiter les fichiers de validation
        if progress_callback:
            progress_callback("Préparation des données de validation...", processed / total_files * 100)
        else:
            print("Préparation des données de validation...")
            
        for xml_file in val_files:
            self._process_file(xml_file, "val", dirs)
            processed += 1
            if progress_callback:
                progress_callback(None, processed / total_files * 100)

        # Créer le fichier de configuration
        self.create_data_yaml()

        summary = f"\nPréparation terminée:\n- Fichiers d'entraînement: {len(train_files)}\n- Fichiers de validation: {len(val_files)}\n- Classes: {sorted(self.classes)}"
        
        if progress_callback:
            progress_callback(summary, 100)
        else:
            print(summary)

    def _process_file(self, xml_file, split_type, dirs):
        """Traite un fichier individuel"""
        # Convertir les annotations
        yolo_lines = self.convert_xml_to_yolo(xml_file)

        # Copier l'image
        img_file = xml_file.with_suffix(".png")
        if img_file.exists():
            shutil.copy2(img_file, dirs[f"images/{split_type}"])

            # Sauvegarder les annotations YOLO
            yolo_file = dirs[f"labels/{split_type}"] / f"{img_file.stem}.txt"
            with open(yolo_file, "w") as f:
                f.write("\n".join(yolo_lines))

    def create_data_yaml(self):
        """Crée le fichier data.yaml pour YOLOv11"""
        data = {
            "path": str(self.dataset_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.classes),
            "names": sorted(list(self.classes)),
        }

        yaml_file = self.dataset_dir / "data.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(data, f, sort_keys=False)


def train_yolo(dataset_path, epochs=100, batch_size=16, img_size=640, progress_callback=None):
    """Lance l'entraînement YOLOv11"""
    try:
        import torch

        if progress_callback:
            gpu_info = f"GPU disponible: {torch.cuda.is_available()}\n"
            if torch.cuda.is_available():
                gpu_info += f"Nom du GPU: {torch.cuda.get_device_name(0)}\n"
                gpu_info += f"CUDA version: {torch.version.cuda}"
            progress_callback(gpu_info, 0)
        else:
            print(torch.cuda.is_available())  # Cela doit retourner True
            print(torch.cuda.get_device_name(0))  # Nom de votre GPU
            print(torch.version.cuda)

        from ultralytics import YOLO

        if progress_callback:
            progress_callback("Configuration de l'entraînement YOLOv11...", 10)
        else:
            print("Configuration de l'entraînement YOLOv11...")

        # Charger le modèle YOLOv11
        model = YOLO("yolo11n.pt")  # ou autre version selon vos besoins

        # Configurer l'entraînement
        training_args = {
            "data": str(Path(dataset_path) / "data.yaml"),
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": img_size,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "workers": 8,
            "patience": 50,
            "save": True,
        }

        # Lancer l'entraînement
        if progress_callback:
            progress_callback("Démarrage de l'entraînement...", 20)
        else:
            print("Démarrage de l'entraînement...")
            
        model.train(**training_args)

        if progress_callback:
            progress_callback("Entraînement terminé!", 100)
        else:
            print("Entraînement terminé!")

    except ImportError as e:
        error_msg = f"Erreur: Veuillez installer les packages nécessaires:\npip install ultralytics torch\n{str(e)}"
        if progress_callback:
            progress_callback(error_msg, -1)
        else:
            print(error_msg)


class YOLOTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv11 Training Interface")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Chemin du fichier de configuration
        self.config_file = Path("config.json")
        
        # Variables par défaut
        self.data_dir = tk.StringVar()
        self.split_ratio = tk.DoubleVar(value=0.2)
        self.epochs = tk.IntVar(value=100)
        self.batch_size = tk.IntVar(value=16)
        self.img_size = tk.IntVar(value=640)
        
        # Charger la configuration si elle existe
        self.load_config()
        
        # Création de l'interface
        self.create_widgets()
        
    def create_widgets(self):
        # Style
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10))
        style.configure("TLabel", font=("Arial", 10))
        style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        
        # Notebook (onglets)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Onglet d'entraînement
        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text="Entraînement")
        
        # Onglet de gestion des classes
        self.classes_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.classes_tab, text="Gestion des Classes")
        
        # Configuration de l'onglet d'entraînement
        self.setup_training_tab()
        
        # Configuration de l'onglet de gestion des classes
        self.setup_classes_tab()
        
    def setup_training_tab(self):
        # Frame principal
        main_frame = ttk.Frame(self.train_tab, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titre
        header = ttk.Label(main_frame, text="Configuration de l'entraînement YOLOv11", style="Header.TLabel")
        header.pack(pady=(0, 20))
        
        # Section de sélection du dossier de données
        data_frame = ttk.LabelFrame(main_frame, text="Dossier de données", padding=10)
        data_frame.pack(fill=tk.X, pady=10)
        
        data_entry = ttk.Entry(data_frame, textvariable=self.data_dir, width=60)
        data_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(data_frame, text="Parcourir", command=self.browse_data_dir)
        browse_btn.pack(side=tk.RIGHT, padx=5)
        
        # Section des paramètres
        params_frame = ttk.LabelFrame(main_frame, text="Paramètres d'entraînement", padding=10)
        params_frame.pack(fill=tk.X, pady=10)
        
        # Grille pour les paramètres
        params_grid = ttk.Frame(params_frame)
        params_grid.pack(fill=tk.X, pady=5)
        
        # Split ratio
        ttk.Label(params_grid, text="Ratio de validation:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        split_scale = ttk.Scale(params_grid, from_=0.1, to=0.5, variable=self.split_ratio, length=200, command=self.on_param_change)
        split_scale.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.split_value_label = ttk.Label(params_grid, text=f"{self.split_ratio.get():.1f}")
        self.split_value_label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Epochs
        ttk.Label(params_grid, text="Nombre d'epochs:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        epochs_entry = ttk.Spinbox(params_grid, from_=1, to=1000, textvariable=self.epochs, width=10, command=self.on_param_change)
        epochs_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        epochs_entry.bind("<KeyRelease>", lambda e: self.on_param_change())
        
        # Batch size
        ttk.Label(params_grid, text="Taille du batch:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        batch_entry = ttk.Spinbox(params_grid, from_=1, to=128, textvariable=self.batch_size, width=10, command=self.on_param_change)
        batch_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        batch_entry.bind("<KeyRelease>", lambda e: self.on_param_change())
        
        # Image size
        ttk.Label(params_grid, text="Taille de l'image:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        img_size_combo = ttk.Combobox(params_grid, textvariable=self.img_size, width=10)
        img_size_combo['values'] = (320, 416, 512, 640, 768, 896, 1024)
        img_size_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        img_size_combo.bind("<<ComboboxSelected>>", lambda e: self.on_param_change())
        
        # Section des actions
        actions_frame = ttk.Frame(main_frame)
        actions_frame.pack(fill=tk.X, pady=10)
        
        prepare_btn = ttk.Button(actions_frame, text="Préparer les données", command=self.prepare_dataset)
        prepare_btn.pack(side=tk.LEFT, padx=5)
        
        train_btn = ttk.Button(actions_frame, text="Lancer l'entraînement", command=self.start_training)
        train_btn.pack(side=tk.LEFT, padx=5)
        
        save_config_btn = ttk.Button(actions_frame, text="Sauvegarder config", command=self.save_config)
        save_config_btn.pack(side=tk.LEFT, padx=5)
        
        # Zone de log
        log_frame = ttk.LabelFrame(main_frame, text="Journal", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Barre de progression
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X)
        
        # Mettre à jour l'affichage du ratio
        self.on_param_change()
    
    def setup_classes_tab(self):
        # Frame principal
        main_frame = ttk.Frame(self.classes_tab, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titre
        header = ttk.Label(main_frame, text="Gestion des classes dans les annotations", style="Header.TLabel")
        header.pack(pady=(0, 20))
        
        # Section de sélection du dossier de données
        data_frame = ttk.LabelFrame(main_frame, text="Dossier de données", padding=10)
        data_frame.pack(fill=tk.X, pady=10)
        
        data_entry = ttk.Entry(data_frame, textvariable=self.data_dir, width=60)
        data_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(data_frame, text="Parcourir", command=self.browse_data_dir)
        browse_btn.pack(side=tk.RIGHT, padx=5)
        
        # Section des actions
        actions_frame = ttk.Frame(main_frame)
        actions_frame.pack(fill=tk.X, pady=10)
        
        list_classes_btn = ttk.Button(actions_frame, text="Lister les classes", command=self.list_classes)
        list_classes_btn.pack(side=tk.LEFT, padx=5)
        
        update_class_btn = ttk.Button(actions_frame, text="Modifier une classe", command=self.update_class)
        update_class_btn.pack(side=tk.LEFT, padx=5)
        
        # Tableau des classes
        classes_frame = ttk.LabelFrame(main_frame, text="Classes détectées", padding=10)
        classes_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Créer un Treeview pour afficher les classes
        columns = ("Classe", "Occurrence")
        self.classes_tree = ttk.Treeview(classes_frame, columns=columns, show="headings")
        
        # Définir les en-têtes de colonnes
        self.classes_tree.heading("Classe", text="Classe")
        self.classes_tree.heading("Occurrence", text="Occurrence")
        
        # Définir les largeurs de colonnes
        self.classes_tree.column("Classe", width=200)
        self.classes_tree.column("Occurrence", width=100)
        
        # Ajouter une barre de défilement
        scrollbar = ttk.Scrollbar(classes_frame, orient=tk.VERTICAL, command=self.classes_tree.yview)
        self.classes_tree.configure(yscrollcommand=scrollbar.set)
        
        # Placer les widgets
        self.classes_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def browse_data_dir(self):
        directory = filedialog.askdirectory(title="Sélectionner le dossier de données")
        if directory:
            self.data_dir.set(directory)
            self.log(f"Dossier sélectionné: {directory}")
            self.save_config()
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_progress(self, message=None, value=None):
        if message:
            self.log(message)
        if value is not None:
            if value < 0:  # Erreur
                self.progress['value'] = 0
            else:
                self.progress['value'] = value
        self.root.update_idletasks()
    
    def on_param_change(self, *args):
        # Mettre à jour l'affichage du ratio
        self.split_value_label.config(text=f"{self.split_ratio.get():.1f}")
        # Sauvegarder automatiquement les paramètres
        self.save_config()
    
    def save_config(self):
        """Sauvegarde la configuration dans un fichier JSON"""
        config = {
            "data_dir": self.data_dir.get(),
            "split_ratio": self.split_ratio.get(),
            "epochs": self.epochs.get(),
            "batch_size": self.batch_size.get(),
            "img_size": self.img_size.get()
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            self.log(f"Configuration sauvegardée dans {self.config_file}")
        except Exception as e:
            self.log(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
    
    def load_config(self):
        """Charge la configuration depuis un fichier JSON"""
        if not self.config_file.exists():
            return
            
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
            # Appliquer les valeurs chargées
            if "data_dir" in config:
                self.data_dir.set(config["data_dir"])
            if "split_ratio" in config:
                self.split_ratio.set(config["split_ratio"])
            if "epochs" in config:
                self.epochs.set(config["epochs"])
            if "batch_size" in config:
                self.batch_size.set(config["batch_size"])
            if "img_size" in config:
                self.img_size.set(config["img_size"])
                
        except Exception as e:
            print(f"Erreur lors du chargement de la configuration: {str(e)}")
    
    def prepare_dataset(self):
        data_dir = self.data_dir.get()
        if not data_dir:
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier de données")
            return
            
        split_ratio = self.split_ratio.get()
        
        self.log(f"Préparation du dataset avec un ratio de validation de {split_ratio:.2f}")
        self.progress['value'] = 0
        
        def run_preparation():
            try:
                preparator = YOLOTrainingPreparator(data_dir)
                preparator.prepare_dataset(split_ratio=split_ratio, progress_callback=self.update_progress)
                self.dataset_dir = preparator.dataset_dir
                # Sauvegarder la configuration après une préparation réussie
                self.save_config()
            except Exception as e:
                self.update_progress(f"Erreur lors de la préparation: {str(e)}", -1)
        
        Thread(target=run_preparation).start()
    
    def start_training(self):
        if not hasattr(self, 'dataset_dir'):
            if not messagebox.askyesno("Attention", "Le dataset n'a pas été préparé. Voulez-vous continuer avec le dossier 'dataset' dans le répertoire sélectionné?"):
                return
            self.dataset_dir = Path(self.data_dir.get()) / "dataset"
        
        epochs = self.epochs.get()
        batch_size = self.batch_size.get()
        img_size = self.img_size.get()
        
        self.log(f"Démarrage de l'entraînement avec {epochs} epochs, batch size {batch_size}, image size {img_size}")
        self.progress['value'] = 0
        
        # Sauvegarder la configuration avant l'entraînement
        self.save_config()
        
        def run_training():
            try:
                train_yolo(
                    dataset_path=self.dataset_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    img_size=img_size,
                    progress_callback=self.update_progress
                )
            except Exception as e:
                self.update_progress(f"Erreur lors de l'entraînement: {str(e)}", -1)
        
        Thread(target=run_training).start()
    
    def list_classes_in_dataset(self, folder_path):
        """
        Parcourt tous les fichiers XML dans un dossier et liste les classes présentes et leur occurrence.
        
        :param folder_path: Chemin vers le dossier contenant les fichiers XML.
        :return: DataFrame des classes et de leur occurrence.
        """
        if not os.path.exists(folder_path):
            self.log(f"Le dossier {folder_path} n'existe pas.")
            return None

        class_counts = {}
        files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]

        if not files:
            self.log("Aucun fichier XML trouvé dans le dossier.")
            return None
            
        self.progress['value'] = 0
        total_files = len(files)
        processed = 0

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
                self.log(f"Erreur lors du traitement du fichier {file}: {e}")
            except Exception as e:
                self.log(f"Une erreur s'est produite avec le fichier {file}: {e}")
                
            processed += 1
            self.progress['value'] = (processed / total_files) * 100
            self.root.update_idletasks()

        # Créer un DataFrame pour afficher les classes et leurs occurrences
        df = pd.DataFrame(list(class_counts.items()), columns=["Classe", "Occurrence"])
        df = df.sort_values(by="Occurrence", ascending=False).reset_index(drop=True)

        return df
    
    def update_class_in_files(self, folder_path, old_class, new_class):
        """
        Met à jour les fichiers XML pour remplacer une classe par une autre.
        
        :param folder_path: Chemin vers le dossier contenant les fichiers XML.
        :param old_class: Nom de la classe à remplacer.
        :param new_class: Nouveau nom de la classe.
        """
        files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
        
        if not files:
            self.log("Aucun fichier XML trouvé dans le dossier.")
            return
            
        self.progress['value'] = 0
        total_files = len(files)
        processed = 0
        updated_files = 0

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
                    updated_files += 1
                    self.log(f"Classe mise à jour dans le fichier {file}")

            except ET.ParseError as e:
                self.log(f"Erreur lors du traitement du fichier {file}: {e}")
            except Exception as e:
                self.log(f"Une erreur s'est produite avec le fichier {file}: {e}")
                
            processed += 1
            self.progress['value'] = (processed / total_files) * 100
            self.root.update_idletasks()
            
        self.log(f"\nMise à jour terminée: {updated_files} fichiers modifiés sur {total_files} fichiers analysés.")
    
    def list_classes(self):
        """Affiche les classes présentes dans les fichiers XML"""
        data_dir = self.data_dir.get()
        if not data_dir:
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier de données")
            return
            
        self.log(f"Analyse des classes dans le dossier {data_dir}...")
        
        def run_analysis():
            try:
                # Vider le treeview
                for item in self.classes_tree.get_children():
                    self.classes_tree.delete(item)
                    
                # Lister les classes
                df_classes = self.list_classes_in_dataset(data_dir)
                
                if df_classes is not None:
                    # Remplir le treeview avec les données
                    for _, row in df_classes.iterrows():
                        self.classes_tree.insert("", "end", values=(row["Classe"], row["Occurrence"]))
                    
                    self.log(f"Analyse terminée: {len(df_classes)} classes trouvées.")
                else:
                    self.log("Aucune classe trouvée.")
                    
            except Exception as e:
                self.log(f"Erreur lors de l'analyse des classes: {str(e)}")
                
        Thread(target=run_analysis).start()
    
    def update_class(self):
        """Met à jour une classe dans les fichiers XML"""
        data_dir = self.data_dir.get()
        if not data_dir:
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier de données")
            return
            
        # Vérifier si des classes ont été listées
        if not self.classes_tree.get_children():
            if not messagebox.askyesno("Attention", "Aucune classe n'a été listée. Voulez-vous d'abord lister les classes?"):
                return
            self.list_classes()
            return
            
        # Sélectionner la classe à modifier
        selected_item = self.classes_tree.selection()
        if not selected_item:
            old_class = simpledialog.askstring("Modifier une classe", "Entrez le nom de la classe à modifier:")
            if not old_class:
                return
        else:
            old_class = self.classes_tree.item(selected_item[0], "values")[0]
            
        # Demander le nouveau nom
        new_class = simpledialog.askstring("Modifier une classe", f"Entrez le nouveau nom pour la classe '{old_class}':")
        if not new_class:
            return
            
        # Confirmation
        if not messagebox.askyesno("Confirmation", f"Voulez-vous remplacer la classe '{old_class}' par '{new_class}' dans tous les fichiers XML?"):
            return
            
        self.log(f"Mise à jour de la classe '{old_class}' vers '{new_class}'...")
        
        def run_update():
            try:
                self.update_class_in_files(data_dir, old_class, new_class)
                # Mettre à jour la liste des classes
                self.list_classes()
            except Exception as e:
                self.log(f"Erreur lors de la mise à jour des classes: {str(e)}")
                
        Thread(target=run_update).start()


if __name__ == "__main__":
    # Lancer l'interface graphique
    root = tk.Tk()
    app = YOLOTrainingGUI(root)
    root.mainloop()
