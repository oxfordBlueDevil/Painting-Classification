from lasagne import layers
from lasagne.nonlinearities import  softmax, rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from pyimage.pipeline import ImagePipeline
from sklearn.metrics import f1_score, classification_report
import numpy as np
import cPickle

if __name__ == '__main__':
	print 'Building Image Pipeline'
	imp = ImagePipeline('../scraped-images')
	imp.read(sub_dirs = ['Cezanne', 'VanGogh', 'JosephMallordTurner'])
	#imp.read(sub_dirs = ['Durer', 'Klimt-and-Expressionism'])
	imp.resize(shape = (360, 360, 3))
	imp.images_to_patches(patch_size=(80,96), max_patches=30)

	imp.vectorize()
	# X = imp.features.astype(np.float32)
	# X_tilda = imp.merge_features_dominant_colors()
	# y = imp.labels.astype(np.int32)
	imp.vectorize(isImage=False)
	X_patches = imp.patch_features.astype(np.float32)
	y_patches = imp.patch_labels.astype(np.int32)

	X_scaled = StandardScaler().fit_transform(X_patches)
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_patches, test_size=0.2, random_state=23)

	nnet = NeuralNet(
	          # Specify the layers
	          layers=[('input', layers.InputLayer),
	                  ('hidden1', layers.DenseLayer),
	                  ('hidden2', layers.DenseLayer),
	                  ('hidden3', layers.DenseLayer),
	                  ('output', layers.DenseLayer)
	          		  ],

	          # Input Layer
	          input_shape=(None, X_scaled.shape[1]),

	          # Hidden Layer 1
	          hidden1_num_units=512,
	          hidden1_nonlinearity=rectify,

	          # Hidden Layer 2
	          hidden2_num_units=512,
	          hidden2_nonlinearity=rectify,

	          # # Hidden Layer 3
	          hidden3_num_units=512,
	          hidden3_nonlinearity=rectify,

	          # Output Layer
	          output_num_units=3,
	          output_nonlinearity=softmax,

	          # Optimization
	          update=nesterov_momentum,
	          update_learning_rate=0.001,
	          update_momentum=0.3,
	          max_epochs=30,

	          # Others,
	          regression=False,
	          verbose=1,
	    )

	# # Train the NN
	print 'Training Neural Network'
	nnet.fit(X_train, y_train)

	print
	# Make predictions
	y_predictions = nnet.predict(X_test)
	print "f1 score:", f1_score(y_test, y_predictions, average='weighted')

	print classification_report(y_test, y_predictions)

	with open(r"../Pickled-Models/three-artist-neural-network.pickle", "wb") as output_file:
		cPickle.dump(nnet, output_file, protocol=cPickle.HIGHEST_PROTOCOL)
