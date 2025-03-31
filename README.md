# YOLOv11 Annotator

Un outil d'annotation automatique basé sur YOLOv11 pour simplifier la labélisation de documents.

![Logo](logo.png)

## Description

YOLOv11 Annotator est une application qui utilise le modèle YOLOv11 pour détecter et annoter automatiquement des éléments dans des documents. L'application permet de:

- Charger et prévisualiser des images
- Détecter automatiquement des objets avec YOLOv11
- Sélectionner et filtrer les prédictions pertinentes
- Générer des annotations au format XML (compatible avec PASCAL VOC)
- Organiser les données annotées dans un dossier dédié

## Fonctionnalités

- Interface graphique intuitive avec Tkinter
- Prévisualisation des détections avec code couleur par classe
- Filtrage des détections par classe et seuil de confiance
- Génération automatique d'UUID pour les fichiers
- Export des annotations au format XML standard

## Prérequis

- Python 3.8+
- PyTorch
- Ultralytics
- OpenCV
- Tkinter
- PIL (Pillow)

## Installation

1. Clonez ce dépôt:
   ```
   git clone https://github.com/Florian0077/yolov11_annotator.git
   cd yolov11_annotator
   ```

2. Créez un environnement virtuel et activez-le:
   ```
   # Avec venv (Python standard)
   python -m venv venv
   
   # Activation sur Windows
   venv\Scripts\activate
   
   # Activation sur Linux/Mac
   source venv/bin/activate
   ```

3. Installez les dépendances:
   ```
   pip install -r requirements.txt
   ```

4. Assurez-vous que votre modèle YOLOv11 est placé dans le dossier `model/` (par défaut `best.pt`)

## Utilisation

1. Exécutez le script `run.bat` ou lancez directement:
   ```
   python app.py
   ```

2. Utilisez l'interface pour:
   - Sélectionner un dossier d'images
   - Ajuster le seuil de confiance si nécessaire
   - Parcourir les images et leurs prédictions
   - Cocher/décocher les prédictions à conserver
   - Valider pour générer les annotations XML

## Configuration

Le fichier `config.json` permet de personnaliser:

- Le chemin du modèle YOLOv11
- Le dossier de sortie pour les annotations
- La liste des noms de classes
- Le seuil de confiance par défaut

## Structure des dossiers

```
yolov11_annotator/
├── app.py              # Application principale
├── config.json         # Configuration du projet
├── logo.png            # Logo de l'application
├── model/              # Dossier contenant le modèle YOLOv11
│   └── best.pt         # Modèle pré-entraîné
├── auto_dataset/       # Dossier de sortie pour les annotations
├── requirements.txt    # Dépendances Python
└── run.bat             # Script de lancement pour Windows
```

## Licence

Ce projet est distribué sous licence [MIT](LICENSE).

## Auteur

[Florian0077](https://github.com/Florian0077)
