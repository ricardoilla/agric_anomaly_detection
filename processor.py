import glob
from xtract_features.glcms import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import cv2
import numpy as np
from sklearn import svm
import warnings
import mahotas
import json
from reader import read
from time import time
warnings.filterwarnings("ignore")


###############################
# FUNCIONES PARA EXTRACCIÓN
# DE FEATURES
###############################

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def fd_haralick(image):  # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def fd_glcm(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feats = glcm(img)
    out = feats.glcm_all()
    return out


def run_detection(img_path):
    tiempo_inicial = time()
    sizes = [25, 20, 15]
    # sizes = [50,40,30]
    print(img_path)
    img = cv2.imread(img_path)

    for SIZE in sizes:
        ##########################################################################
        # SEPARO SUELO DE VEGETACION CON EL HISTOGRAMA DE COLORES y CLUSTERING k=2
        ##########################################################################
        output = {}
        winW = SIZE
        winH = SIZE
        stride = SIZE
        feature_matrix = []
        clone = img
        xmax, ymax, _ = img.shape
        for x in range(0, xmax, SIZE):
            for y in range(0, ymax, SIZE):
                new_img = img[x:x+SIZE, y:y+SIZE]
                if new_img.shape ==(SIZE,SIZE,3):
                    global_features = np.hstack([fd_histogram(new_img)])
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    global_features = global_features.reshape(-1, 1)
                    rescaled_features = scaler.fit_transform(global_features)
                    feature_matrix.append(rescaled_features)

        X_train = np.asarray(feature_matrix)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X_train)
        y_kmeans = kmeans.predict(X_train)
        centers = kmeans.cluster_centers_
        max1 = np.argmax(centers[0])
        max2 = np.argmax(centers[1])

        ##########################################################################
        # CREO LISTA DE CUADROS CON VEGETACION
        ##########################################################################
        seg0 = []
        seg1 = []
        if max1 > max2:
            plant_cluster = True
        else:
            plant_cluster = False
        clone = img
        xmax, ymax, _ = img.shape
        iter = 0
        for x in range(0, xmax, stride):
            for y in range(0, ymax, stride):
                distances = []
                new_img = img[x:x + SIZE, y:y + SIZE]
                if new_img.shape == (SIZE, SIZE, 3):
                    try:
                        clust = y_kmeans[iter]
                        if plant_cluster:
                            if clust == 0:
                                seg0.append(iter)
                                pass
                            elif clust == 1:
                                seg1.append(iter)
                                cv2.rectangle(clone, (y, x), (y + winW, x + winH), (0, 0, 0), cv2.FILLED)
                        else:
                            if clust == 0:
                                seg1.append(iter)
                                cv2.rectangle(clone, (y, x), (y + winW, x + winH), (0, 0, 0), cv2.FILLED)
                            elif clust == 1:
                                seg0.append(iter)
                                pass
                    except:
                        print('iter Error')
                    iter += 1
        img=cv2.imread(img_path)
        # cv2.imshow('Clone', clone)
        # cv2.imshow('imgA', img)
        # cv2.waitKey(0)
        dst = cv2.addWeighted(clone, 0.5, img, 0.5, 0)
        cv2.imwrite('temp/temp_img.jpg', dst)
        rule=seg0
        ########################################################################
        # UTILIZO SVM ONE CLASS Y GLCM PARA DETECTAR ANOMALÍAS DE TEXTURA
        # SOLO EN LOS CUADROS CON VEGETACIÓN
        ########################################################################

        clf = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.98)
        feature_matrix=[]
        largo = 48
        clone = img
        xmax, ymax, _ = img.shape
        count = 0
        for x in range(0, xmax, SIZE):
            for y in range(0, ymax, SIZE):
                new_img = img[x:x+SIZE, y:y+SIZE]
                name = str(int(x / SIZE)) + '_' + str(int(y / SIZE))
                if new_img.shape == (SIZE, SIZE, 3):
                    if count in rule:
                        output[name] = []
                        features = fd_glcm(new_img)
                        largo = features.shape[0]
                        for value in features:
                            output[name].append(str(value))
                    count += 1
        with open('features_glcm.json', 'w') as fp:
            json.dump(output, fp)
        X_train2, windows = read('features_glcm.json', features_len=largo)

        clf.fit(X_train2)
        predictions = clf.predict(X_train2)

        # Muestro los resultados:
        img2 = cv2.imread('temp/temp_img.jpg')
        clone = img2
        xmax, ymax, _ = img2.shape
        count = 0
        it = 0
        for x in range(0, xmax, SIZE):
            for y in range(0, ymax, SIZE):
                new_img = img2[x:x + SIZE, y:y + SIZE]
                if new_img.shape == (SIZE, SIZE, 3):
                    if count in rule:
                        try:
                            id = predictions[it]
                            it += 1
                            if id == 1:
                                cv2.rectangle(clone, (y, x), (y + winW, x + winH), (0, 0, 255), cv2.FILLED)
                        except IndexError as e:
                            print(e)
                    count += 1
        res = cv2.imread('temp/temp_img.jpg')
        dst = cv2.addWeighted(clone, 0.6, res, 0.4, 0)
        cv2.imwrite('temp/Result'+str(SIZE)+'.jpg', dst)

    img_list = []
    for SIZE in sizes:
        img_list.append(cv2.imread('temp/Result'+str(SIZE)+'.jpg'))
    for x in range(len(img_list)):
        if x==0:
            dst = img_list[0]
        else:
            dst = cv2.addWeighted(dst, 0.5, img_list[x], 0.5, 0)

    tiempo_final = time()
    tiempo_ejecucion = tiempo_final - tiempo_inicial
    print('El tiempo de ejecucion fue:', tiempo_ejecucion)
    return dst