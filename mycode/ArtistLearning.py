from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import f1_score
from pipeline import ImagePipeline
import cPickle

def previewImages(imp):
	imp.show('Durer', 1)
	imp.show('Durer', 3)
	imp.show('Durer', 5)
	imp.show('VanGogh', 145)
	imp.show('VanGogh', 134)
	imp.show('VanGogh', 56)
	imp.show('JosephMallordTurner', 183)
	imp.show('JosephMallordTurner', 1)
	imp.show('JosephMallordTurner', 79)

def printEstimatorScores(model, X_test, y_test):
	y_predict = model.predict(X_test)
	print "F1 Score:", f1_score(y_test, y_predict, average='weighted')
	print

def printF1CVScore(scores):
	print "F1 CV Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
	print

def svm(X, y, cv=5):
    svm = SVC(C=1.5)
    print 'Support Vector Machine Cross Validation'
    f1_scores = cross_val_score(svm, X, y, cv=cv, scoring='f1', n_jobs=-1)
    print svm
    printF1CVScore(f1_scores)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=23)
    print 'Support Vector Machine is Fitting'
    svm.fit(X_train, y_train)
    printEstimatorScores(svm, X_test, y_test)
    return svm

def random_forest(X, y, cv=5):
    rfModel = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    f1_scores = cross_val_score(rfModel, X, y, cv=cv, scoring='f1', n_jobs=-1)
    print rfModel
    printF1CVScore(f1_scores)
    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.2, random_state=23)
    print 'RandomForestClassifier is Fitting'
    rfModel.fit(X_train, y_train)
    printEstimatorScores(rfModel, X_test, y_test)
    return rfModel

def gradient_boosting(X, y, cv=5):
	gbModel = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
	f1_scores = cross_val_score(gbModel, X, y, cv=cv, scoring='f1', n_jobs=-1)
	print gbModel
	printF1CVScore(f1_scores)
	print 'GradientBoostingClassifier is Fitting'
	X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.2, random_state=23)
	gbModel.fit(X_train, y_train)
	printEstimatorScores(gbModel, X_test, y_test)
	return gbModel

def knn(X, y, cv=5):
    knn = KNeighborsClassifier()
    f1_scores = cross_val_score(knn, X, y, cv=cv, scoring='f1', n_jobs=-1)
    print knn
    printF1CVScore(f1_scores)
    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.2, random_state=23)
    print 'k Nearest Neighbors is Fitting'
    knn.fit(X_train, y_train)
    printEstimatorScores(knn, X_test, y_test)
    return knn

if __name__ == '__main__':
	print 'Building Image Pipeline'
	imp = ImagePipeline('../scraped-images')
	imp.read(sub_dirs = ['Cezanne', 'VanGogh', 'JosephMallordTurner'])
	#imp.read(sub_dirs = ['Durer', 'Klimt-and-Expressionism'])
	imp.resize(shape = (480, 480, 3))
	imp.images_to_patches()
	imp.patches_to_dominant_colors(n_clusters=3)
	#imp.images_to_dominant_colors(n_clusters=3)

	#Dummy Modeling with Grayscaling, Desnoising and Canny Edge Detection
	imp.vectorize()
	X = imp.features
	# X_tilda = imp.merge_features_dominant_colors()
	y = imp.labels

	#Preprocessing Patches with Grayscaling, Desnoising and Canny Edge Detection
	imp.grayscale(isImage=False)
	imp.tv_denoise(weight = 0.15, isImage=False)
	imp.canny(sigma=1.5, isImage=False)
	imp.vectorize(isImage=False)

	X_patches = imp.patch_features
	X_patches_hat = imp.merge_features_dominant_colors(isImage=False)
	y_patches = imp.patch_labels
	
	#Run Dummy Models
	print 'Running Dummy Models'
	svModel = svm(X_patches_hat, y, cv=5)
	rfModel = random_forest(X_patches_hat, y_patches, cv=5)
	gbModel = gradient_boosting(X_patches_hat, y_patches, cv=5)
	knnModel = knn(X_patches_hat, y_patches, cv=5)

	with open(r"Three-Painter-GradientBoost.pickle", "wb") as output_file:
		cPickle.dump(gbModel, output_file)

