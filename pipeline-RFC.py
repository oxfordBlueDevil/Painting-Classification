from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from pyimage.pipeline import ImagePipeline

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

def printScore(model, X_train, X_test, y_train, y_test):
	train_accuracy = model.score(X_train, y_train)
	test_accuracy = model.score(X_test, y_test)
	print model
	print 'The training accuracy of the Random Forest classifier is: ', train_accuracy
	print 'The test accuracy of the Random Forest classifier is: ', test_accuracy
	print

if __name__ == '__main__':
	imp = ImagePipeline('scraped-images')
	imp.read(sub_dirs = ['Durer', 'JosephMallordTurner', 'VanGogh'])
	# previewImages(imp)
	imp.resize(shape = (640, 640, 3))
	# previewImages(imp)
	imp.vectorize()

	#Dummy Model
	X1 = imp.features
	y1 = imp.labels
	X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.8)
	RF_model_dummy = RandomForestClassifier(n_estimators = 10000, n_jobs=-1,  min_samples_leaf=2)
	RF_model_dummy.fit(X1_train, y1_train)
	printScore(RF_model_dummy, X1_train, X1_test, y1_train, y1_test)
	# del RF_model_dummy

	# SVM_dummy = SVC(C = 1.5)
	# SVM_dummy.fit(X1_train, y1_train)
	# printScore(SVM_dummy, X1_train, X1_test, y1_train, y1_test)

	#Dummy Model with Desnoising
	imp.grayscale()
	# previewImages(imp)
	#imp.denoise_bilateral()
	#imp.tv_denoise()
	imp.canny(sigma=2.25, sub_dir = ['VanGogh'])
	imp.canny(sigma=1.1, sub_dir = ['Durer', 'JosephMallordTurner'])
	# imp.sobel()
	previewImages(imp)
	imp.vectorize()
	X2 = imp.features
	y2 = imp.labels
	X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.8)
	RF_model2 = RandomForestClassifier(n_estimators = 10000, min_samples_leaf = 2, n_jobs=-1)
	RF_model2.fit(X2_train, y2_train)
	printScore(RF_model2, X2_train, X2_test, y2_train, y2_test)

	# SVM_model2 = SVC(C = 1.5)
	# SVM_model2.fit(X2_train, y2_train)
	# printScore(SVM_model2, X2_train, X2_test, y2_train, y2_test)

	# imp.tv_denoise(weight = 1.2)
	# previewImages(imp)
	# imp.vectorize()
	# X3 = imp.features
	# y3 = imp.labels
	# X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size = 0.8)
	# RF_model3 = RandomForestClassifier(n_jobs=-1)
	# RF_model3.fit(X3_train, y3_train)
	# printScore(RF_model3, X3_train, X3_test, y3_train, y3_test)
