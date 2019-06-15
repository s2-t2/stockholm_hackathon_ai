from pprint import pprint
from time import time
import logging

import pandas

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

csv = pandas.read_csv("data/spooky-author-identification/train.csv")
data = list(csv.text)
target = list(csv.author)

testcsv = pandas.read_csv("data/spooky-author-identification/test.csv")
testdata = list(testcsv.text)
testid = list(testcsv.id)

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(tol=1e-3)),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.75,),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 5),),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__max_iter': (20,),
    'clf__alpha': (1e-7),
    #'clf__penalty': ('l2', 'elasticnet'),
    'clf__loss': ('log',),
    'clf__max_iter': (160,),
}

if __name__ == "__main__":
    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, cv=5,
                               n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data, target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    output = grid_search.predict_proba(testdata)
    result = pandas.DataFrame(output, columns=grid_search.classes_)

    id = pandas.DataFrame(testid, columns =["id"])
    res = id.join(result).set_index('id')

    res.to_csv("result.csv")




    





