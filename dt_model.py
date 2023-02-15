
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset from a CSV file
df = pd.read_csv("Iris.csv")

# Prepare the input and target data
# input = df['Species']
target = df['Species']
le_Species = LabelEncoder()
target_encoded = le_Species.fit_transform(target)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']], target_encoded, test_size=0.2, random_state=42)

# Create a decision tree classifier model
model = tree.DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, Y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model on the testing data
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

import pickle
# saving the model a pickle file
# pickle.dump(tree.DecisionTreeClassifier(),open('DT_model.pkl','wb'))
with open('DT_model.pkl','wb') as file:
    pickle.dump(model,file)

# loading the model to disk
# pickle.dump(tree.DecisionTreeClassifier(),'DT_model.pkl','rb')
with open('DT_model.pkl','rb') as file:
    mode= pickle.load(file)

# visualising the dataset

# iris=pd.read_csv("Iris.csv")
# # print(iris.head())

# fig,ax=plt.subplots()
# ax.set_title(' Iris Dataset ')
# ax.set_xlabel('SepalLengthCm')
# ax.set_ylabel('SepalWidthCm')
# colors={'Iris-setosa':'r','Iris-versicolor':'b','Iris-virginica':'g'}
# for i in range(len(iris['SepalLengthCm'])):
#     ax.scatter(iris['SepalLengthCm'][i],iris['SepalWidthCm'][i],
#     color=colors[iris['Species'][i]])
# ax.legend()
# plt.show()


# Visualising using seaborn 
# import seaborn as sns

# sns.set_style("whitegrid")
# sns.FacetGrid(iris,hue="Species",
#                 height=6).map(plt.scatter,
#                 'SepalLengthCm',
#                 'SepalWidthCm').add_legend()
# plt.show()