from nolearn.lasagne import NeuralNet
from sklearn.metrics import f1_score
from skimage import io, transform
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import StandardScaler
import numpy as np

class preProcessUploadedImage(object):
	def __init__(self, filename, model, isArtist=True):
		self.filename = filename
		self.model = model
		self.isArtist = isArtist

	def preprocessing(self):
		img_arr = io.imread(self.filename)
		self.img_arr = transform.resize(img_arr, (360, 360, 3), preserve_range=True)
		patches = extract_patches_2d(img_arr, patch_size=(80,96), max_patches=30)
		self.patches = list(patches)
		self.n_patches = 30

	def vectorize(self):	
		row_tup = tuple(patch_arr.ravel()[np.newaxis, :] for patch_arr in self.patches)
		X_test = np.r_[row_tup].astype(np.float32)
		self.X_test = StandardScaler().fit_transform(X_test)

	def majorityVote(self, predictions, isArtist=True):
		count_arr = []
		labels = [0, 1, 2]
		for label in labels:
			mask = predictions == label
			count = np.sum(mask)/float(predictions.shape[0])
			count_arr.append(count)
		if isArtist:
			return self.labelToArtist(np.argmax(count_arr))
		else:
			return self.labelToStyle(np.argmax(count_arr))

	def labelToArtist(self, label):
		artists = ['Cezanne', 'Van Gogh', 'Joseph Mallord Turner']
		labels = [0, 1, 2]
		d = {label: artist for (label, artist) in zip(labels, artists)}
		return d[label]

	def labelToStyle(self, label):
		artists = ['Portrait', 'Not A Portrait']
		labels = [0, 1]
		d = {label: artist for (label, artist) in zip(labels, artists)}
		return d[label]

	def predict(self):
		predictions = self.model.predict(self.X_test)
		return self.majorityVote(predictions, isArtist=self.isArtist)

