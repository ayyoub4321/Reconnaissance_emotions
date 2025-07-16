# Système de Détection d'Émotions Faciales
## Description
Ce projet implémente un système de détection d'émotions faciales utilisant des techniques de vision par ordinateur et d'apprentissage automatique. Le système analyse les expressions faciales dans des images pour prédire l'émotion correspondante.
## Émotions Détectées
Le système peut détecter 6 émotions principales :

Ahegao 
Angry  
Happy  
Neutral 
Sad 
Surprise 

## Technologies Utilisées

Python - Langage principal
OpenCV - Traitement d'images
MediaPipe - Détection de points faciaux
scikit-learn - Apprentissage automatique
XGBoost - Modèle de prédiction
Django - Interface web
Pandas - Manipulation de données
NumPy - Calculs numériques

## Structure du Projet
```
detecteur/
├── models.py              # Modèles Django
├── views.py               # Vues Django
├── excraNewData.py        # Extraction des caractéristiques
├── pointKey.py            # Points clés faciaux (MediaPipe)
├── filter.py              # Filtres d'images
├── modelsML/              # Modèles ML entraînés
│   └── Best_XGBoostV2.pkl
├── resultatModel/         # Résultats des modèles
│   └── Resultats_Models2V2.csv
└── templates/             # Templates HTML
    ├── index.html
    ├── display.html
    ├── models.html
    └── ...
    ```
