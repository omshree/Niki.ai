from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd 
#Path to the labeled data
path = "data.txt"
#Loading the labeled data to an array
with open("data.txt") as f:
	data = f.read()
	data = data.split("\n")
	data = [x.split(",,,") for x in data]
	for row in data:
		if len(row)!=2:
			data.remove(row)
	data = [[x[0].strip(),x[1].strip()] for x in data]

#Loading the data array into a Pandas DataFrame and mapping the labels to a numerical value (0-unknown, 1-what,2-who,3-affirmation,when-4)
data = np.array(data)
data_df = pd.DataFrame(data,columns=["text","label"])
data_df['label'] = data_df.label.map({'what':1,'who':2,'affirmation':3,'when':4,'unknown':0})
print (data_df.head(10))
#Splitiing target and input variables
X = data_df.text
y = data_df.label
#Spliting into Test and Train Data Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
#Building a Document Term Matrix
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

#Applying Naive Bayes (MultiNomial) Classifier
nb = MultinomialNB()
#Training the Classifier
nb.fit(X_train_dtm, y_train)
#Evaluating the test Data
y_pred_class = nb.predict(X_test_dtm)
test_custom = ["When will you be here?"]
test_dtm = vect.transform(test_custom)
print (nb.predict(test_dtm))

#Calculating the Accuracy of the Model on Test Data
print (metrics.accuracy_score(y_test, y_pred_class))