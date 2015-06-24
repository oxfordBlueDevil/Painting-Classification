from nolearn.lasagne import NeuralNet
from sklearn.metrics import f1_score
from skimage import io, transform
from sklearn.feature_extraction.image import extract_patches_2d
import cPickle as pickle
import numpy as np

class ArtistClassifier(object):
	def __init__(self):
		with open(r"Pickled-Models/three-artist-neural-network.pickle", "rb") as nn:
			self.nn = pickle.load(nn)
	
	def predict(self, X_test):
		return self.nn.predict(X_test)

	def f1(self, y_test, y_predictions):
		return f1_score(y_test, y_predictions)

if __name__ == '__main__':
	nn = ArtistClassifier()
	Keeper = io.imread('scraped-images/VanGogh/Keeper-of-lunatic-asylum-of-Saint-Paul--Trabucby-Van-Gogh.jpg')
	Keeper = transform.resize(Keeper, (360, 360, 3), preserve_range=True)
	patches = extract_patches_2d(Keeper, patch_size=(80,96), max_patches=30)
	patches = list(patches)
	row_tup = tuple(patch_arr.ravel()[np.newaxis, :] for patch_arr in patches)
	#Vectorize
	X_test = np.r_[row_tup].astype(np.float32)
	print nn.predict(X_test)