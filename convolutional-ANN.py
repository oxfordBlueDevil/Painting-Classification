from lasagne import layers
from lasgne.init import Constant, Normal
from lasagne.nonlinearities import  softmax, rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from pyimage.pipeline import ImagePipeline
from sklearn.metrics import f1_score, classification_report
import numpy as np
import cPickle

class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb

class AdjustVariable(object):
    def __init__(self, name, start=0.01, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn):
		valid_loss = np.array([i["valid_loss"] for i in nn.train_history_])
		if len(valid_loss) > 1:
			delta = valid_loss[-1] - valid_loss[-2]
			if delta <= 1.0e-3:
			    new_value = np.cast['float32'](self.start/10.0)
			    getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

if __name__ == '__main__':
	print 'Building Image Pipeline'
	imp = ImagePipeline('scraped-images')
	imp.read(sub_dirs = ['Cezanne', 'VanGogh', 'JosephMallordTurner'])
	#imp.read(sub_dirs = ['Durer', 'Klimt-and-Expressionism'])
	imp.resize(shape = (256, 256, 3))
	imp.images_to_patches(patch_size=(224,224), max_patches=10)

	imp.vectorize(isImage=False)
	X_patches = imp.patch_features.astype(np.float32)
	y_patches = imp.patch_labels.astype(np.int32)

	X_scaled = StandardScaler().fit_transform(X_patches)
	X_scaled = X_scaled.reshape(-1, 3, 360, 360)
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_patches, test_size=0.2, random_state=23)

	nnet = NeuralNet(
	    layers=[
	        ('input', layers.InputLayer),
	        ('conv1', layers.Conv2DLayer),
	        ('pool1', layers.MaxPool2DLayer),
	        ('conv2', layers.Conv2DLayer),
	        ('pool2', layers.MaxPool2DLayer),
	        ('conv3', layers.Conv2DLayer),
	        ('conv4', layers.Conv2DLayer),
	        ('conv5', layers.Conv2DLayer),
	        ('pool3', layers.MaxPool2DLayer),
	        ('dropout1',layers.DropoutLayer),
	        ('hidden1', layers.DenseLayer),
	        ('dropout2',layers.DropoutLayer),
	        ('hidden2', layers.DenseLayer),
	        ('output', layers.DenseLayer),
	        ],
	    input_shape=(None, 3, 224, 224),
	    conv1_num_filters=96, conv1_filter_size=(11, 11), conv1_W=Normal(std=0.01, mean=0), conv1_stride=4,
	    pool1_pool_size=(2, 2),
	    conv2_num_filters=256, conv2_filter_size=(2, 2), conv2_W=Normal(std=0.01, mean=0), conv2_b=Constant(val=1.0), 
	    pool2_pool_size=(2, 2),
	    conv3_num_filters=384, conv3_filter_size=(2, 2), conv3_W=Normal(std=0.01, mean=0),
	    conv4_num_filters=384, conv4_filter_size=(2, 2), conv4_W=Normal(std=0.01, mean=0), conv4_b=Constant(val=1.0),
	    conv5_num_filters=256, conv5_filter_size=(2, 2), conv5_W=Normal(std=0.01, mean=0), conv5_b=Constant(val=1.0),
	    pool3_pool_size=(2,2),
	    
	    #Dropout Layer 1
        dropout1_p=0.5,

	    #Hidden Layer 4
	    hidden4_num_units=4096,
	    hidden4_W=Normal(std=0.01, mean=0),
	    hidden4_b=Constant(val=1.0),
	    hidden4_nonlinearity=rectify,

	    #Dropout Layer 2
        dropout2_p=0.5,

	    #Hidden Layer 5
	    hidden5_num_units=4096,
	    hidden5_W=Normal(std=0.01, mean=0),
	    hidden5_b=Constant(val=1.0),
	    hidden5_nonlinearity=rectify,
	
	    #Output Layer
	    output_num_units=3, 
	    output_nonlinearity=softmax,

	    #Optimization
	    update=nesterov_momentum,
	    update_learning_rate=0.01,
	    update_momentum=0.9,
	    max_epochs=90,


	    regression=False,
	    batch_iterator_train=FlipBatchIterator(batch_size=128),
    	on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01),
        EarlyStopping(patience=200),
        ],
	    verbose=1,

	    )
	nnet.fit(X_train, y_train)

	print
	# Make predictions
	y_predictions = nnet.predict(X_test)
	print "f1 score:", f1_score(y_test, y_predictions, average='weighted')

	print classification_report(y_test, y_predictions)

	with open(r"three-artist-convolutional-neuralNet.pickle", "wb") as output_file:
		cPickle.dump(nnet, output_file)