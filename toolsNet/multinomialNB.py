#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
CLASIFICADOR NAIVE BAYES MULTINOMIAL
De acuerdo con la lectura en el libro, el clasificador multinomial NB se ajusta mejor al propósito de contar palabras
dentro de una cadena de caracteres. Otorga mayor nivel de aciertos respecto al modelo GaussianNB.

'''
#
#---------------------------------------------INICIO---------------------------------------------#
#
#Carga de librerías y funciones
'''
sklearn.naive_bayes.MultinomialNB
Llamada al clasificador naïve Bayes multinomial, efectivo en la clasificación de textos orientados al conteo de
grupos grandes de palabras.
Más información: http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB

sklearn.feature_extraction.text.CountVectorizer
Estructura que convierte una colección de documentos de texto en una matriz de tokens contabilizados.
Más información: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer

sklearn.pipeline.Pipeline
Permite el empaquetado de información para la vectorización de los archivos de entrenamiento y el entrenamiento 
del clasificador, todo en una sola linea
Más información: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

#from sklearn.pipeline import Pipeline

sklearn.feature_extraction.text.TfidfTransformer
Más información: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer

sklearn.datasets.load_files
Esta funcion nos permitirá construir un conjunto de caracteristicas y etiquetas a partir de archivos de texto
para la vectorización y a su vez para el proceso de entrenamiento del clasificador. La estructura de esta
entrada es, cada subdirectorio del directorio raíz hace referencia a una etiqueta de clase,y a su vez cada
documento dentro de los subdirectorios contienen el conjunto de archivos que serán las caracteristicas de la clase.
Mas información: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html
'''
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import load_files
#
#---------------------------------------------MAIN---------------------------------------------#
#
def initClass():
	#Cargar los documentos de entrada
	rawDocs = load_files('toolsNet/sentimentalDataset')
	#Crear el vector de tokenizacion
	countVec = CountVectorizer()
	#Establecer la tokenizacion de la entrada
	trainData = countVec.fit_transform(rawDocs.data)
	#Crear y entrenar el clasificador
	return MultinomialNB().fit(trainData, rawDocs.target), countVec, rawDocs.target_names, countVec.build_analyzer()
#
#---------------------------------------------FIN---------------------------------------------#
#