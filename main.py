#!/usr/bin/env python
# -*- coding: utf-8 -*-
#No existe variante switch case en python. Posible aproximación: http://bytebaker.com/2008/11/03/switch-case-statement-in-python/

from toolsNet import multinomialNB as mnb
from toolsNet import neuralNet as ann

classif, countVec, classes, analyze = mnb.initClass()

#Entradas
introText = [
'adoro tu caricia junto a tu beso',
'me dio mucha risa escuchar ese chiste tan bueno y la carcajada de la risa fue excelente, yo lo adoro y lo quiero',
'ese golpe y esa patada no lo pudieron detener',
'Quiero darte un golpe con fuerza',
'Estoy muy frustrado y tengo miedo al fracaso'
]

#Llevar a token la entrada y predecir su clasificación
testDataCount = countVec.transform(introText)
predict = classif.predict(testDataCount)

#Mostrar el resultado de la clasificación
for doc, category in zip(introText, predict):
	print (countVec.transform([doc]).toarray())
	print ('ID de categoría: ' + str(category) + '\n' + str(doc) + ' => ' + str(classes[category]))

#FIN
print ('Success')