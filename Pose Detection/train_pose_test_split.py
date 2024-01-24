import pandas as pd
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 


#Open csv file
df = pd.read_csv('coords.csv')

#Prints head and tail of csv file
df.head()
df.tail()

# DataFrame containing only the rows where the class is 'Happy'.
df[df['class'] == 'Happy']

#For X matrix, we drop the classifier
X = df.drop('class', axis=1)

#For y result, only take feature (Happy)
y = df['class']


#30% testing, 70% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()), #linear model for binary classification.
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()), #regularized linear regression variant.
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()), #fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging 
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()), #builds a sequence of weak learners (typically decision trees) and combines them to form a strong model.
}

fit_models = {}

#loop through pipelines
for algo, pipeline in pipelines.items():
    #train model using train data
    model = pipeline.fit(X_train, y_train)
    #store under models
    fit_models[algo] = model

#loop through models
for algo, model in fit_models.items():
    #predict y hat value
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))

with open('body_language.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)