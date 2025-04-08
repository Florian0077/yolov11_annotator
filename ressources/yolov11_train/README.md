# YOLOv11 Training Interface

Une interface graphique pour faciliter l'entraînement de modèles YOLOv11 à partir d'annotations XML.

## Fonctionnalités

- Interface graphique intuitive basée sur Tkinter
- Conversion automatique des annotations XML au format YOLO
- Configuration facile des paramètres d'entraînement
- Sauvegarde et chargement automatique de la configuration
- Suivi en temps réel de la progression
- Gestion asynchrone des tâches longues

## Prérequis

- Python 3.8+
- PyTorch
- Ultralytics (YOLOv11)
- Tkinter

Consultez le fichier `requirements.txt` pour la liste complète des dépendances.

## Installation

1. Clonez ce dépôt ou téléchargez les fichiers
2. Installez les dépendances :

```bash
pip install -r requirements.txt
```

## Utilisation

1. Exécutez le script principal :

```bash
python yolov11_train.py
```

2. Dans l'interface :
   - Sélectionnez le dossier contenant vos images et annotations XML
   - Configurez les paramètres d'entraînement (ratio de validation, epochs, batch size, etc.)
   - Cliquez sur "Préparer les données" pour convertir les annotations
   - Cliquez sur "Lancer l'entraînement" pour démarrer l'entraînement

## Structure des données

Le programme s'attend à trouver des fichiers d'images (.png) et leurs annotations correspondantes (.xml) dans le dossier sélectionné. Les annotations doivent être au format XML (style Pascal VOC).

## Configuration

Les paramètres sont automatiquement sauvegardés dans un fichier `config.json` et rechargés au démarrage de l'application.

## Résultats

Après l'entraînement, les résultats (modèles, graphiques, etc.) sont sauvegardés dans le dossier `runs/train` créé par YOLOv11.
