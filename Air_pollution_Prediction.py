import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

td = pd.read_csv(r'D:\AI MAFIA\Assignment 2\Training Data\Train.csv')
ted = pd.read_csv(r'D:\AI MAFIA\Assignment 2\Testing Data\Test.csv')

tx = td.drop(['target'], axis =1)
ty = td['target']

print(tx.shape,ty.shape)

print("Training values:\n\n",tx)
print("\n------------------------------------------------------------------------------------\n")
print("Training Dataset Output Values:\n\n",ty)

print("\n------------------------------------------------------------------------------------\n")
model = LinearRegression()

print("Visualisation of the training Data set:")
print("\n------------------------------------------------------------------------------------\n")
plt.scatter(tx['feature_1'], ty,s=10,label = 'feature_1')
plt.scatter(tx['feature_2'], ty,s=10,label = 'feature_2')
plt.scatter(tx['feature_3'], ty,s=10,label = 'feature_3')
plt.scatter(tx['feature_4'], ty,s=10,label = 'feature_4')
plt.scatter(tx['feature_5'], ty,s=10,label = 'feature_5')
plt.legend()
plt.show()

#training model
print("The model is being trained:\n")
print("...\n")
print("...\n")
print("...\n")
print("...\n")
print("...\n")
print("...\n")
model.fit(tx,ty)
print("Model has been successfully trained !!")
print("\n---------------------------------------------------------------------------------  \n")
print('The accuracy of the model is :', model.score(tx,ty))

output = model.predict(ted)
#predicted table
prt = ted
prt['target_predicted'] = output

print("Predicted Values according to trained model and test values :\n")

print("\n------------------------------------------------------------------------------------\n")
print(prt)

# Visualisation
print("Plotting the predicted outputs and orignal outputs\n")

print("\n--------------------------------------------------------------------------------------------\n")
plt.figure(figsize = (10,5))
plt.scatter(tx['feature_1'], ty,s=10,label = 'feature_1')
plt.scatter(tx['feature_2'], ty,s=10,label = 'feature_2')
plt.scatter(tx['feature_3'], ty,s=10,label = 'feature_3')
plt.scatter(tx['feature_4'], ty,s=10,label = 'feature_4')
plt.scatter(tx['feature_5'], ty,s=10,label = 'feature_5')
plt.plot(ted['feature_1'], output, label = 'Predicted')
plt.plot(ted['feature_2'], output, label = 'Predicted')
plt.plot(ted['feature_3'], output, label = 'Predicted')
plt.plot(ted['feature_4'], output, label = 'Predicted')
plt.plot(ted['feature_5'], output, label = 'Predicted')
plt.legend()
plt.show()


