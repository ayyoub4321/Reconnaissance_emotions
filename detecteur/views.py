# views.py

import cv2
import os
import joblib
import numpy as np
import pandas as pd
import ast
from django.conf import settings
from .pointKey import *
from .excraNewData import *
from sklearn.model_selection import train_test_split

from django.shortcuts import render, redirect
from .models import UploadedImage  

EMOTION_MAP2 = {
    0: 'Ahegao', 1: 'Angry', 2: 'Happy',
    3: 'Neutral', 4: 'Sad', 5: 'Surprise'
}
EMOTION_MAPImog = {
    0: '😎',  
    1: '😤',   
    2: '🥰',   
    3: '😐',   
    4: '🥹',   
    5: '🤯',  
}

def index(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('image')
        hidden_value = request.POST.get('hidden')
        
        if uploaded_file:
            image = UploadedImage(image=uploaded_file)
            ext=image.image.name.split('.')[-1]
            image.image.name=f'img{lenght()}.{ext}'
            image.save()
            return redirect('charger_image', image_id=image.id)
    return render(request,'index.html')
def display_modele(request):
    data=load_data()
    data['MatriceConfusion'] = data['MatriceConfusion'].apply(ast.literal_eval)
    modeles = data.to_dict(orient='records')
    return render(request, 'models.html', {'modeles': modeles})
           
def resume(request):
    return render(request, 'resume.html')
def display_image(request, image_id):
    image = UploadedImage.objects.get(id=image_id)
    image_path = os.path.join(settings.MEDIA_ROOT, str(image.image))
    
    image_path = image_path.replace('\\', '/')
    img=cv2.imread(image_path)
    em,emg=predict_image(img,r'detecteur\modelsML\Best_XGBoostV2.pkl')
    
    dossier_Point = os.path.join(settings.MEDIA_ROOT, 'pointKey')
    if not os.path.exists(dossier_Point):
                os.makedirs(dossier_Point)
    x={}
    x['emoji']=emg
    x['image']=image
    x['predict']=em
    return render(request, 'display.html',x)
def lenght(dossier=settings.MEDIA_ROOT):
    dossier=os.listdir(dossier)
    return len(dossier)
# Lire csv de resultat 
def load_data(path=r'detecteur\resultatModel\Resultats_Models2V2.csv'):
    return pd.read_csv(path,sep=',')
    
#Predection

def predict_image(img1, model_path,target_size=(700,600)):

    img = cv2.resize(img1, target_size)
    lm=landmarkPoint(img)
    feat = extractionFutures(img,lm)
    if feat is None:
        return "Impossible d'extraire les features de l'image.", "❌"
    feat = np.array(feat).reshape(1, -1)

    clf = joblib.load(model_path)
    pred = clf.predict(feat)[0]
    # ici
    filename = os.path.basename(model_path)
    filename = filename[5:-4]
    # ici
    em=EMOTION_MAP2.get(pred, "Unknown")
    emog=EMOTION_MAPImog.get(pred,'ma3arfch')
    return em ,emog
def modelsExpli(request):
    return render(request,'ML.html')
def rapport(request):
    return render(request,"rapport.html")