# facial_features.py
"""
Module de fonctions pour extraire et tracer des caractéristiques faciales avec MediaPipe Face Mesh.
"""
import cv2
import numpy as np
import time
from mediapipe.python.solutions.face_mesh import FaceMesh
import math

# ---- Fonction principale ---- #
def P(lm, idx, size):
    """Convertit le landmark normalisé en coordonnées entières."""
    x, y = lm[idx].x * size[0], lm[idx].y * size[1]
    return int(x), int(y)


def landmarkPoint(img):
    """Détection des points faciaux avec MediaPipe"""
    res = mp_face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    return res.multi_face_landmarks[0].landmark

# Initialisation de FaceMesh
mp_face_mesh = FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Fonctions utilitaires
def midpoint(p1, p2):
    """Retourne le milieu entre p1 et p2."""
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def distance(frame, p1, p2, color=(255, 0, 0)):
    """Trace un segment p1-p2 sur frame et renvoie la distance euclidienne."""
    cv2.line(frame, p1, p2, color, 2)
    return np.linalg.norm(np.array(p1) - np.array(p2))
def angle(p1, p2):
    """Renvoie l’angle (en radians) entre p1→p2."""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def extract_face(img, lm):
    """Extraction précise du visage avec gestion des marges"""
    h, w = img.shape[:2]
    # Indices des points du contour facial
    face_outline_idx = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109
    ]
    
    points = np.array([P(lm, idx, (w, h)) for idx in face_outline_idx], dtype=np.int32)
    
    # Création du masque facial avec Convex Hull
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Extraction et recadrage du visage
    face_only = cv2.bitwise_and(img, img, mask=mask)
    x, y, w_face, h_face = cv2.boundingRect(points)
    
    # Marges avec vérification des limites
    margin = 10
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(w, x + w_face + margin)
    y_end = min(h, y + h_face + margin)
    
    return face_only[y_start:y_end, x_start:x_end]

def extract_landmarks(frame, lm):
    """Extraction des points clés avec vérification des dimensions"""
    h, w = frame.shape[:2]
    
    try:
        pts = {
            'mouth_left': P(lm, 61, (w, h)),
            'mouth_right': P(lm, 291, (w, h)),
            'mouth_top': P(lm, 13, (w, h)),
            'mouth_bottom': P(lm, 14, (w, h)),
            'eyeL_outer': P(lm, 33, (w, h)),
            'eyeL_inner': P(lm, 133, (w, h)),
            'eyeR_outer': P(lm, 362, (w, h)),
            'eyeR_inner': P(lm, 263, (w, h)),
            'browL_int': P(lm, 65, (w, h)),
            'browL_ext': P(lm, 52, (w, h)),
            'browR_int': P(lm, 295, (w, h)),
            'browR_ext': P(lm, 285, (w, h)),
            'eyeL_top': P(lm, 159, (w, h)),
            'eyeL_bottom': P(lm, 145, (w, h)),
            'eyeR_top': P(lm, 386, (w, h)),
            'eyeR_bottom': P(lm, 374, (w, h))
        }
    except IndexError as e:
        print(f"Erreur d'indice: {str(e)}")
        return None
    
    pts.update({
        'eyeL_center': midpoint(pts['eyeL_outer'], pts['eyeL_inner']),
        'eyeR_center': midpoint(pts['eyeR_outer'], pts['eyeR_inner']),
        'browL_center': midpoint(pts['browL_int'], pts['browL_ext']),
        'browR_center': midpoint(pts['browR_int'], pts['browR_ext'])
    })
    
    return pts

def bouch(face, landmark , margin = [0, 0, 0, 0] , Trace=False):
    points =  [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    coords = [P(landmark, i, (face.shape[1], face.shape[0])) for i in points]
    x_min = int(min(c[0] for c in coords))
    x_max = int(max(c[0] for c in coords))
    y_min = int(min(c[1] for c in coords))
    y_max = int(max(c[1] for c in coords))
    if Trace:
        cv2.rectangle(face, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    return face[y_min-margin[0]:y_max+margin[2],x_min-margin[3]:x_max+margin[1]]

def nez(face, landmark , margin = [0, 0, 0, 0] , Trace=False):
    points = [1, 2, 98, 327, 168]
    coords = [P(landmark, i, (face.shape[1], face.shape[0])) for i in points]
    x_min = int(min(c[0] for c in coords))
    x_max = int(max(c[0] for c in coords))
    y_min = int(min(c[1] for c in coords))
    y_max = int(max(c[1] for c in coords))
    if Trace:
        cv2.rectangle(face, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    return face[y_min-margin[0]:y_max+margin[2],x_min-margin[3]:x_max+margin[1]]

def eyeDroit(face, landmark , margin = [0, 0, 0, 0] , Trace=False):
    # Correction des indices pour l'œil droit
    points = [362, 263, 386, 374, 385, 380, 387, 373]
    coords = [P(landmark, i, (face.shape[1], face.shape[0])) for i in points]
    x_min = int(min(c[0] for c in coords))
    x_max = int(max(c[0] for c in coords))
    y_min = int(min(c[1] for c in coords))
    y_max = int(max(c[1] for c in coords))
    if Trace:
        cv2.rectangle(face, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    return face[y_min-margin[0]:y_max+margin[2],x_min-margin[3]:x_max+margin[1]]

def eyeGauche(face, landmark , margin = [0, 0, 0, 0] , Trace=False):
    # Correction des indices pour l'œil gauche
    points = [33, 133, 159, 145, 158, 153, 144, 160]
    coords = [P(landmark, i, (face.shape[1], face.shape[0])) for i in points]
    x_min = int(min(c[0] for c in coords))
    x_max = int(max(c[0] for c in coords))
    y_min = int(min(c[1] for c in coords))
    y_max = int(max(c[1] for c in coords))
    if Trace:
        cv2.rectangle(face, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    return face[y_min-margin[0]:y_max+margin[2],x_min-margin[3]:x_max+margin[1]]

def browDroit(face, landmark , margin = [0, 0, 0, 0] , Trace=False):
    points = [46,53,52,65,55]
    coords = [P(landmark, i, (face.shape[1], face.shape[0])) for i in points]
    x_min = int(min(c[0] for c in coords))
    x_max = int(max(c[0] for c in coords))
    y_min = int(min(c[1] for c in coords))
    y_max = int(max(c[1] for c in coords))
    if Trace:
        cv2.rectangle(face, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    return face[y_min-margin[0]:y_max+margin[2],x_min-margin[3]:x_max+margin[1]]

def browGauche(face, landmark, margin = [0, 0, 0, 0] , Trace=False ):
    points = [276,283,282,295,285]
    coords = [P(landmark, i, (face.shape[1], face.shape[0])) for i in points]
    x_min = int(min(c[0] for c in coords))
    x_max = int(max(c[0] for c in coords))
    y_min = int(min(c[1] for c in coords))
    y_max = int(max(c[1] for c in coords))
    if Trace:
        cv2.rectangle(face, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    return face[y_min-margin[0]:y_max+margin[2],x_min-margin[3]:x_max+margin[1]]