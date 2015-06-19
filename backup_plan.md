##Back_up Plan

### Portrait vs. Landscape Model
* To test whether we can build an painting image detection model, we decided to experiment with Random Forest, Gradient Boosting, Support Vector Machine, K Nearest Neighbor Classifiers to determine whether an image is a Portrait painting or a Landscape painting. 

* We obtained the following F1 scores by cross-validation where K = 8:
	1.) RandomForestClassifier(n_estimators=1000, oob_score=True):
		a.) F1 Score: 0.65 (+/- 0.03)
	2.) GradientBoostingClassifier(learning_rate=0.1, n_estimators=100):
		a.) F1 Score: 0.62 (+/- 0.02)
	3.) KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=5, p=2, weights='uniform'):
        a.) F1 Score: 0.59 (+/- 0.00)

* After cross-validation and experimentation, the Random Forest Classifier where number of estimators is 1000 proved to have highest the F1 score of the classifiers we tested. Thus, we've chosen this RandomForestClassifier Model for our baseline model to determine whether whether an image is a Portrait painting or a Landscape painting. For now, we look to take on a much challenging problem.

* The plan for this project is to build an Artist Classification model. So far the performance of our baseline model gives us hope that we can build a multilabeled Classifier to detect which artist made whichever painting.