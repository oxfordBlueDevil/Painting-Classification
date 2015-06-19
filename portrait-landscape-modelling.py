from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from pyimage.pipeline import ImagePipeline
import cPickle

def printF1CVScore(scores):
	print "F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
	print

def svm(X, y, cv=8):
    svm = SVC(C=1.5)
    f1_scores = cross_val_score(svm, X, y, cv=cv, scoring='f1', n_jobs=-1)
    print svm
    printF1CVScore(f1_scores)
    return svm

def random_forest(X, y, cv=8):
    rfModel = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    f1_scores = cross_val_score(rfModel, X, y, cv=cv, scoring='f1', n_jobs=-1)
    print rfModel
    printF1CVScore(f1_scores)
    # print "rf precision:", precision_score(y_test, y_predict)
    # print "rf recall:", recall_score(y_test, y_predict)
    return rfModel

def gradient_boosting(X, y, cv=8):
	gbModel = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
	f1_scores = cross_val_score(gbModel, X, y, cv=cv, scoring='f1', n_jobs=-1)
	print gbModel
	printF1CVScore(f1_scores)
	return gbModel

def knn(X, y, cv=8):
    knn = KNeighborsClassifier()
    f1_scores = cross_val_score(knn, X, y, cv=cv, scoring='f1', n_jobs=-1)
    print knn
    printF1CVScore(f1_scores)
    return knn

def printScore(model, X_train, X_test, y_train, y_test):
	train_accuracy = model.score(X_train, y_train)
	test_accuracy = model.score(X_test, y_test)
	print model
	print 'The training F1 score of the model is: ', train_accuracy
	print 'The test F1 score of the model is: ', test_accuracy
	print

if __name__ == '__main__':
	print 'Building Image Pipeline'
	imp = ImagePipeline('scraped-images')
	imp.read(sub_dirs = ['Portraits', 'Landscapes'])
	imp.resize(shape = (480, 480, 3))
	imp.images_to_dominant_colors(n_clusters=3)

	#Dummy Modeling with Grayscaling, Desnoising and Canny Edge Detection
	imp.grayscale()
	#imp.denoise_bilateral()
	imp.tv_denoise(weight = 0.15)
	imp.canny(sigma=1.5)
	# imp.sobel()
	imp.vectorize()
	X = imp.features
	X_tilda = imp.merge_features_dominant_colors()
	y = imp.labels
	
	#Run Dummy Models
	print 'Running Dummy Models'
	svModel = svm(X_tilda, y, cv=8)
	rfModel = random_forest(X_tilda,y, cv=8)
	gbModel = gradient_boosting(X_tilda, y, cv=8)
	knnModel = knn(X_tilda, y, cv=8)

	# with open(r"image-pipeline2.pickle", "wb") as output_file:
	# 	cPickle.dump(imp, output_file)
