import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create random training and testing data sets
df = pd.read_csv('preprocessed_data.csv')
training_data, test_data = train_test_split(df, test_size=0.2, random_state=0)

# Create pipeline
pipeline = Pipeline(
    [
        ("vect", TfidfVectorizer()),
        ("clf", DecisionTreeClassifier()),
    ]
)
pipeline

# Parameters to test
parameter_grid = {
    "vect__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),
    "vect__ngram_range": ((1, 1), (1, 2)),
    "vect__binary": (True, False),
    "clf__criterion": ('gini', 'entropy', 'log_loss'),
    "clf__max_depth": (None, 3, 6),
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