from .pointKey import extract_landmarks,distance,angle,extract_face,bouch,browDroit,browGauche,eyeDroit,eyeGauche,nez,landmarkPoint

from .filter import (gabor_filter,hog_filter,laplacian_filter,sobel_filter,lbp_filter
                    
                    
                    )
import pandas as pd
import os
import cv2
import numpy as np
def CreatStructeurData(lbp_bins=2):
    """
    Crée la structure de données avec un nombre configurable de features LBP
    
    Args:
        lbp_bins (int): Nombre de bins pour l'histogramme LBP (défaut=26)
    """
    df = pd.DataFrame(columns=[
    # Filtres par région (20 colonnes)
    "Gabor_Bouche", "HOG_Bouche", "Laplacian_Bouche", "Sobel_Bouche",
    "Gabor_Nez", "HOG_Nez", "Laplacian_Nez", "Sobel_Nez",
    "Gabor_EyeDroit", "HOG_EyeDroit", "Laplacian_EyeDroit", "Sobel_EyeDroit",
    "Gabor_BrowDroit", "HOG_BrowDroit", "Laplacian_BrowDroit", "Sobel_BrowDroit",
    "LBP_Bouche", "LBP_Nez", "LBP_EyeDroit", "LBP_BrowDroit",

    # Distances faciales (20 colonnes)
    "Distance_Mouth_Left_Right", "Distance_Mouth_Top_Bottom",
    "Distance_EyeR_Outer_Inner", "Distance_EyeR_Top_Bottom",
    "Distance_EyeL_Outer_Inner", "Distance_EyeL_Top_Bottom",
    "Distance_EyeR_Center_BrowR_Center", "Distance_EyeL_Center_BrowL_Center",
    "Distance_Mouth_Right_EyeR_Center", "Distance_Mouth_Left_EyeL_Center",
    "Distance_Mouth_Top_EyeL_Center", "Distance_Mouth_Top_EyeR_Center",
    "Distance_BrowL_Center_BrowR_Center", "Distance_Mouth_Bottom_BrowL_Center",
    "Distance_Mouth_Bottom_BrowR_Center", "Distance_EyeL_Center_EyeR_Center",
    "Distance_EyeL_Inner_EyeR_Inner", "Distance_EyeL_Outer_BrowL_Ext",
    "Distance_EyeR_Outer_BrowR_Ext", "Distance_EyeL_Top_BrowL_Int",

    # Ratios et aire (2 colonnes)
    "Ratio_Opening_Bouche", "Aire_EyeL",

    # Angles (5 colonnes)
    "Angle_BrowL_Int_Ext", "Angle_BrowR_Int_Ext",
    "Angle_Mouth_Left_Right", "Angle_EyeL_Top_Bottom",
    "Angle_EyeR_Top_Bottom"
    
])
    return df 
    
    

def extractionFutures(frame,lm):
    if lm is None:
            return None
    
    frame=extract_face(frame,lm)
    print("===========")
    print(frame.shape)
    print("===========")
    
    frame=cv2.resize(frame,(300,400))
    lm=landmarkPoint(frame)
    if lm is None:
        return None
    pts = extract_landmarks(frame,lm)
    if pts is None:
        return None 
    zonBouche=bouch(frame,lm,[20,10,20,10])
    zonNez=nez(frame,lm,[20,10,20,10])
    zonEyeDroit=eyeDroit(frame,lm,[10,10,10,10])
    zonBrowDroit=browDroit(frame,lm,[10,10,10,10])
    mesures = [
            #  Filtres par région (20 colonnes
            gabor_filter(zonBouche),        # "Gabor_Bouche"
            hog_filter(zonBouche),          # "HOG_Bouche"
            laplacian_filter(zonBouche),    # "Laplacian_Bouche"
            sobel_filter(zonBouche),        # "Sobel_Bouche"
            
            gabor_filter(zonNez),           # "Gabor_Nez"
            hog_filter(zonNez),             # "HOG_Nez"
            laplacian_filter(zonNez),       # "Laplacian_Nez"
            sobel_filter(zonNez),           # "Sobel_Nez"
            
            gabor_filter(zonEyeDroit),      # "Gabor_EyeDroit"
            hog_filter(zonEyeDroit),        # "HOG_EyeDroit"
            laplacian_filter(zonEyeDroit),  # "Laplacian_EyeDroit"
            sobel_filter(zonEyeDroit),      # "Sobel_EyeDroit"
            
            gabor_filter(zonBrowDroit),     # "Gabor_BrowDroit"
            hog_filter(zonBrowDroit),       # "HOG_BrowDroit"
            laplacian_filter(zonBrowDroit), # "Laplacian_BrowDroit"
            sobel_filter(zonBrowDroit),     # "Sobel_BrowDroit"
            
            lbp_filter(zonBouche),          # "LBP_Bouche"
            lbp_filter(zonNez),             # "LBP_Nez"
            lbp_filter(zonEyeDroit),        # "LBP_EyeDroit"
            lbp_filter(zonBrowDroit),       # "LBP_BrowDroit"    
                    
            # Distances faciales (20 colonnes)
            distance(frame,pts['mouth_left'], pts['mouth_right']),          # "Distance_Mouth_Left_Right"
            distance(frame,pts['mouth_top'], pts['mouth_bottom']),          # "Distance_Mouth_Top_Bottom"
            distance(frame,pts['eyeR_outer'], pts['eyeR_inner']),           # "Distance_EyeR_Outer_Inner"
            distance(frame,pts['eyeR_top'], pts['eyeR_bottom']),            # "Distance_EyeR_Top_Bottom"
            distance(frame,pts['eyeL_outer'], pts['eyeL_inner']),           # "Distance_EyeL_Outer_Inner"
            distance(frame,pts['eyeL_top'], pts['eyeL_bottom']),            # "Distance_EyeL_Top_Bottom"
            distance(frame,pts['eyeR_center'], pts['browR_center']),        # "Distance_EyeR_Center_BrowR_Center"
            distance(frame,pts['eyeL_center'], pts['browL_center']),        # "Distance_EyeL_Center_BrowL_Center"
            distance(frame,pts['mouth_right'], pts['eyeR_center']),         # "Distance_Mouth_Right_EyeR_Center"
            distance(frame,pts['mouth_left'], pts['eyeL_center']),          # "Distance_Mouth_Left_EyeL_Center"
            distance(frame,pts['mouth_top'], pts['eyeL_center']),           # "Distance_Mouth_Top_EyeL_Center"
            distance(frame,pts['mouth_top'], pts['eyeR_center']),           # "Distance_Mouth_Top_EyeR_Center"
            distance(frame,pts['browL_center'], pts['browR_center']),       # "Distance_BrowL_Center_BrowR_Center"
            distance(frame,pts['mouth_bottom'], pts['browL_center']),       # "Distance_Mouth_Bottom_BrowL_Center"
            distance(frame,pts['mouth_bottom'], pts['browR_center']),       # "Distance_Mouth_Bottom_BrowR_Center"
            distance(frame,pts['eyeL_center'], pts['eyeR_center']),         # "Distance_EyeL_Center_EyeR_Center"
            distance(frame,pts['eyeL_inner'], pts['eyeR_inner']),           # "Distance_EyeL_Inner_EyeR_Inner"
            distance(frame,pts['eyeL_outer'], pts['browL_ext']),            # "Distance_EyeL_Outer_BrowL_Ext"
            distance(frame,pts['eyeR_outer'], pts['browR_ext']),            # "Distance_EyeR_Outer_BrowR_Ext"
            distance(frame,pts['eyeL_top'], pts['browL_int']),               # "Distance_EyeL_Top_BrowL_Int"
            # Ratios et aire (2 colonnes)
            (distance(frame,pts['mouth_top'], pts['mouth_bottom']) / distance(frame,pts['mouth_left'], pts['mouth_right'])),  # "Ratio_Opening_Bouche"
            
            (distance(frame,pts['eyeL_top'], pts['eyeL_bottom']) * distance(frame,pts['eyeL_outer'], pts['eyeL_inner'])),    # "Aire_EyeL"
            # Angles (5 colonnes)
            angle(pts['browL_int'], pts['browL_ext']),   # "Angle_BrowL_Int_Ext"
            angle(pts['browR_int'], pts['browR_ext']),   # "Angle_BrowR_Int_Ext"
            angle(pts['mouth_left'], pts['mouth_right']),# "Angle_Mouth_Left_Right"
            angle(pts['eyeL_top'], pts['eyeL_bottom']),  # "Angle_EyeL_Top_Bottom"
            angle(pts['eyeR_top'], pts['eyeR_bottom'])    # "Angle_EyeR_Top_Bottom"

    ]

    cv2.imshow('face',frame)
    cv2.waitKey(1)
    return mesures

def chargerImage(path,nbImg=200000):
    lable=0
    dic={}
    totale=0
    target=[]
    data=CreatStructeurData()
    for dossier in os.listdir(path):
        nbImgDossier=0
        dic[lable]= dossier
        print(lable,' ==> ',dossier)
        
        chemin_dossier=os.path.join(path, dossier)
        for fichier in os.listdir(chemin_dossier):
                if nbImgDossier>=nbImg:
                    break
                totale+=1
                chemin_image = os.path.join(chemin_dossier, fichier)
                
                img=cv2.imread(chemin_image)
                lm=landmarkPoint(img)
                if lm is None:
                    continue
                
                mesures=extractionFutures(img,lm)
                if 0xFF==ord('q'):
                     print('stop')
                     break
                if mesures is not None:
                    data.loc[len(data)] =mesures
                    target.append(lable)
                    nbImgDossier += 1
                if mesures is None:
                    cv2.imshow('Erreur eci ', img)
                    cv2.waitKey(1)
                    # supprimer_image(chemin_image)
                    continue
        print("nombre des image dans ",dossier,'est ',nbImgDossier)
        lable+=1
    print('nombre toltale des image est ',totale)
    data.head(10)
    data['emotion']=target
    print(f"dectionner est {dic}")
    data.to_csv('Data/dataPath2.csv', index=False)


