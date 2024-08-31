import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create random training and testing data sets
df = pd.read_csv('preprocessed_data.csv')
training_data, test_data = train_test_split(df, test_size=0.2, random_state=0)

# Dummy classifier to calculate random baseline was run once, saved it here as a comment
"""
dummy = DummyClassifier(strategy='uniform', random_state=0)
dummy.fit(training_data['Lemmatized Lyrics'], training_data['Decade'])
dummy.predict(test_data['Lemmatized Lyrics'])
print(dummy.score(test_data['Lemmatized Lyrics'], test_data['Decade']))
"""

# Create pipeline
pipeline = Pipeline(
    [
        ("vect", TfidfVectorizer()),
        ("clf", MultinomialNB()),
    ]
)
pipeline

# Parameters to test
parameter_grid = {
    "vect__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),
    "vect__ngram_range": ((1, 1), (1, 2)),
    "vect__binary": (True, False),
    "clf__alpha": (0.1, 1),
}

# Perform grid search to find best parameters
grid_search = GridSearchCV(estimator=pipeline, param_grid=parameter_grid, n_jobs=2, verbose=1)
grid_search.fit(training_data['Lemmatized Lyrics'], training_data['Decade'])
print('The best parameters were: ' + str(grid_search.best_params_))

# Show results from the best model
true = test_data['Decade']
pred = grid_search.predict(test_data['Lemmatized Lyrics'])
print(classification_report(true, pred))
ConfusionMatrixDisplay.from_predictions(true, pred)
plt.show()