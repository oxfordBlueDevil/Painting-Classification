from nolearn.lasagne import NeuralNet
from sklearn.metrics import f1_score
from skimage import io, transform
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import StandardScaler
from pipeline import ImagePipeline
import cPickle as pickle
import numpy as np
import random

class ArtistClassifier(object):
	def __init__(self):
		"""
		Unpicles trained neural network
		"""
		with open(r"Pickled-Models/three-artist-neural-network.pickle", "rb") as nn:
			self.nn = pickle.load(nn)
	
	def predict(self, X_test):
		"""
		Perform clasification on samples in X_test

		:param X_test: test samples
		"""
		return self.nn.predict(X_test)

	def f1(self, y_test, y_predictions):
		"""
		Compute the F1 Score

		:param y_test: Correct Target Values
		:param y_predictions: Estimated targets as returned by classifier
		"""
		return f1_score(y_test, y_predictions)

class PortraitClassifier(object):
	def __init__(self):
		"""
		Unpicles trained neural network
		"""
		with open(r"Pickled-Models/portrait-neural-network.pickle", "rb") as nn:
			self.nn = pickle.load(nn)
	
	def predict(self, X_test):
		"""
		Perform clasification on samples in X_test

		:param X_test: test samples
		"""
		return self.nn.predict(X_test)

	def f1(self, y_test, y_predictions):
		"""
		Compute the F1 Score

		:param y_test: Correct Target Values
		:param y_predictions: Estimated targets as returned by classifier
		"""
		return f1_score(y_test, y_predictions)

def majorityVote(predictions, isArtist=True):
	"""
	Perform a vote on the classifications of the patches of test image

	:param isArtist: If True, then use artist label. If False, then use genre label.
	:return 
	"""
	count_arr = []
	labels = [0, 1, 2]
	for label in labels:
		mask = predictions == label
		count = np.sum(mask)/float(predictions.shape[0])
		count_arr.append(count)
	if isArtist:
		return labelToArtist(np.argmax(count_arr))
	else:
		return labelToGenre(np.argmax(count_arr))

def labelToArtist(label):
	artists = ['Cezanne', 'Van Gogh', 'Joseph Mallord Turner']
	labels = [0, 1, 2]
	d = {label: artist for (label, artist) in zip(labels, artists)}
	return d[label]

def labelToGenre(label):
	artists = ['Portrait', 'Not A Portrait']
	labels = [0, 1]
	d = {label: artist for (label, artist) in zip(labels, artists)}
	return d[label]


if __name__ == '__main__':
	nn = ArtistClassifier()
	print 'Building Test Image Pipeline'
	imp = ImagePipeline('../scraped-images')
	imp.read(sub_dirs = ['Cezanne'])
	imp.img_lst2[0] = random.sample(imp.img_lst2[0], 100)
	imp.resize(shape = (360, 360, 3))
	imp.images_to_patches(patch_size=(80,96), max_patches=30)
	imp.vectorize(isImage=False)
	X_patches = imp.patch_features.astype(np.float32)
	labels = np.ones((X_patches.shape[0],)).astype(np.int32)*0
	X_scaled = StandardScaler().fit_transform(X_patches)
	predictions = nn.predict(X_scaled)
	predictions_mapped = []
	labels_mapped = []
	for i in xrange(0,100):
		start = i * imp.n_patches
		end = start + imp.n_patches
		labels_mapped.append(majorityVote(labels[start:end]))
		predictions_mapped.append(majorityVote(predictions[start:end]))

	print '# of times predicted Cezanne:', (predictions == 0).sum()/float(predictions.shape[0])
	print '# of times predicted Van Gogh:', (predictions == 1).sum()/float(predictions.shape[0])
	print '# of times predicted Joseph Mallor Turner:', (predictions == 2).sum()/float(predictions.shape[0])
	print
	print labels_mapped[:10]
	print
	print predictions_mapped[:10]
	print

	nn = PortraitClassifier()
	print 'Building Test Image Pipeline'
	imp = ImagePipeline('../scraped-images')
	imp.read(sub_dirs = ['Portraits'])
	imp.img_lst2[0] = random.sample(imp.img_lst2[0], 100)
	imp.resize(shape = (360, 360, 3))
	imp.images_to_patches(patch_size=(80,96), max_patches=30)
	imp.vectorize(isImage=False)
	X_patches = imp.patch_features.astype(np.float32)
	labels = np.ones((X_patches.shape[0],)).astype(np.int32)*0

	X_scaled = StandardScaler().fit_transform(X_patches)
	predictions = nn.predict(X_scaled)
	predictions_mapped = []
	labels_mapped = []
	for i in xrange(0,100):
		start = i * imp.n_patches
		end = start + imp.n_patches
		labels_mapped.append(majorityVote(labels[start:end], isArtist=False))
		predictions_mapped.append(majorityVote(predictions[start:end], isArtist=False))

	print '# of times predicted Portrait:', (predictions == 0).sum()/float(predictions.shape[0])
	print '# of times predicted Not A Portrait:', (predictions == 1).sum()/float(predictions.shape[0])
	print
	print labels_mapped[:10]
	print
	print predictions_mapped[:10]
