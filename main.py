#!/usr/bin/env python
# -*- coding: utf-8 -*-
#No existe variante switch case en python. Posible aproximación: http://bytebaker.com/2008/11/03/switch-case-statement-in-python/

"""
Combinación de red y clasificador en una primera aproximación

Digitos de los sentimientos:
0 - Alegria
1 - Amor
2 - Odio
3 - Tristeza
"""

from toolsNet import multinomialNB as mnb
from toolsNet import neuralNet as ann

classif, countVec, classes, analyze = mnb.initClass()

#Entradas
introText = [
'LLORAR INTRANQUILO PENA GOLPE',
'ESPOSO ESPOSA SEXO AMAR',
'AMOR RISA GOZO JOCOSO',
'AGRESIVO EMPUJAR DESTRUIR BARBARO',
'LLORO ENAMORADO TERNURA NOVIO'
]

#Llevar a token la entrada y predecir su clasificación
testDataCount = countVec.transform(introText)
predict = classif.predict(testDataCount)

#Mostrar el resultado de la clasificación
for doc, category in zip(introText, predict):
	#print (countVec.transform([doc]).toarray())
	print ('ID de categoría: ' + str(category) + '\n' + str(doc) + ' => ' + str(classes[category]))

lvInput = ann.np.array([[1, 1, 1, 2], [3, 3, 1, 0], [2, 1, 0, 0], [2, 0, 2, 2]])
lvTarget = ann.np.array([[1], [-1], [1], [0]])
lFuncs = [None, ann.tanh, ann.sgm]

bpn = ann.BackPropagationNetwork((4,3,1), lFuncs)

lnMax = 50000
lnErr = 1e-5
for i in range(lnMax + 1):
	err = bpn.TrainEpoch(lvInput, lvTarget, momentum = 0.7)
	if i % 2500 == 0:
		print ("Iteration {0}\tError: {1:0.6f}".format(i, err))
	if err <= lnErr:
		print ("Minimum error reached at iteration {0}".format(i))
		break
		
# Display Output
lvOutput = bpn.Run(lvInput)
for i in range(lvInput.shape[0]):
	print ("Input: {0} OutPut: {1}".format(lvInput[i],lvOutput[i]))

#FIN
print ('Success')