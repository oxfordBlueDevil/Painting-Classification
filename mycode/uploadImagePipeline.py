from nolearn.lasagne import NeuralNet
from sklearn.metrics import f1_score
from skimage import io, transform
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import StandardScaler
import numpy as np

class preProcessUploadedImage(object):
	def __init__(self, filename, model, isArtist=True):
		"""
		Manages reading image and loading a classifier to perform a classifcation on the image

		:param filename: image filename
		:param model: classifier
		:param isArtist: If True, then we are using the artist classifier. 
		If False, then we are using the genre classifier. 
		"""
		self.filename = filename
		self.model = model
		self.isArtist = isArtist

	def preprocessing(self):
		"""
		Resize uploaded image and extract patches from it
		"""
		img_arr = io.imread(self.filename)
		self.img_arr = transform.resize(img_arr, (360, 360, 3), preserve_range=True)
		patches = extract_patches_2d(img_arr, patch_size=(80,96), max_patches=30)
		self.patches = list(patches)
		self.n_patches = 30

	def vectorize(self):
		"""
		Take a list of patches and vectorize all the patches. Then generate a
		feature matrix where each row represents a patch of the uploaded image
		"""
		row_tup = tuple(patch_arr.ravel()[np.newaxis, :] for patch_arr in self.patches)
		X_test = np.r_[row_tup].astype(np.float32)
		self.X_test = StandardScaler().fit_transform(X_test)

	def majorityVote(self, predictions, isArtist=True):
		"""
		Perform a vote on the classifications of the patches of test image

		:param isArtist: If True, then use artist label. If False, then use genre label.
		:return label
		"""
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
		"""
		Convert numeric label to corresponding artist label

		:param label: numeric label
		:return artist label
		"""
		artists = ['Cezanne', 'Van Gogh', 'Joseph Mallord Turner']
		labels = [0, 1, 2]
		d = {label: artist for (label, artist) in zip(labels, artists)}
		return d[label]

	def labelToStyle(self, label):
		"""
		Convert numeric label to correspond genre label

		:param label: numeric label
		:return genre label
		"""
		artists = ['Portrait', 'Not A Portrait']
		labels = [0, 1]
		d = {label: artist for (label, artist) in zip(labels, artists)}
		return d[label]

	def predict(self):
		"""
		Perform clasifications on patches and then vote to determine the final classification
		"""
		predictions = self.model.predict(self.X_test)
		return self.majorityVote(predictions, isArtist=self.isArtist)

