###Mike Osorio, Galvanize DSI (fka Zipfian Academy), 4/13/2015 - 07/10/2015

##neuralArt.io
* This is my Data Science capstone project for the Galvanize DSI. I was first drawn by the idea of teaching machines how to detect and categorize objects and scenes in images and videos. The idea of teaching a machine how to perform large-scale fine art painting classification and be capable to make semantic-level judgements, such as predicting a painting's genre (portrait or landscape) and its artist captured my curiousity about computer vision and motivated me to take on this challenge as my capstone project. 

-------  ------------  ------------  -----------  -----------  ------
##Code Walk-through

###Preprocessing

####scrapingImages.py
* This module scrapes high-resolution paintings for different European Artists from a Picasa Web Album to create our dataset. 
The downloaded image resolutions are around 800 x 800. The script saves all paintings in artists subdirectories in a master directoy called scraped-images.

####mycode.pipeline.py
* This module allows us to build our image pipeline and creates the capability to perform a variety of image transformation filters for our EDA & Feature extraction phase.
    1. Denoise Bilateral Denoising
    2. Total Variance Denoising
    3. Resizing All images at once
    3. Grayscaling Transformation
    4. Canny edge detector
    5. Sobel edge detector

* We also created an implemenation for dominant color exctrations. We use mini batch k means to extract the most dominant colors for each image/patch in our image pipeline.

* Furthermore, we implented patch extraction for each of our images in our image pipeline. By extracting random patches, we can increase our training set immensly. Without this scheme, our supervised learning models would suffer from substantial overfitting.

-------  ------------  ------------  -----------  -----------  ------

### Portrait vs. Landscape  (portrait-landscape-modelling.py & portrait-Neural-Network.py)
* Before building our supervised learning models, we preprocessed our images. We resized our images to 480 x 480, extracted 30 random 80 x 96 patches from the 480 x 480 images, calculated the dominant colors for each patch, grayscaled them, applied total variation denoising, and implemented canny edge detection. 

* To test whether we can build an painting image detection model, we decided to experiment with Random Forest, Gradient Boosting, Support Vector Machine, and K Nearest Neighbor Classifiers to determine whether an image is a Portrait painting or a Landscape painting. 

* We obtained the following F1 scores by cross-validation where K = 8:
	- RandomForestClassifier(n_estimators=1000, oob_score=True):
		- F1 Score: 0.65 (+/- 0.03)
	- GradientBoostingClassifier(learning_rate=0.1, n_estimators=100):
		- F1 Score: 0.62 (+/- 0.02)
	- KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=5, p=2, weights='uniform'):
		- F1 Score: 0.59 (+/- 0.00)

  * After cross-validation and experimentation, the Random Forest Classifier where number of estimators is 1000 proved to have highest the F1 score of the non-deep learning classifiers we tested. Thus, we've chosen this RandomForestClassifier Model for our baseline model to determine whether whether an image is a Portrait painting or a Landscape painting. For now, we look to take on a much challenging problem.

* For our deep learning, we only had to resize our images to 360 x 360, extracted 30 random 80 x 96 patches from the 360 x 360 images for our data preprocessing. After preprocessing, we built a Neural Network with three hidden layers which each have 512 outputs. Here are the results:

  - input                 (None, 23040)           produces   23040 outputs
  - hidden1               (None, 512)             produces     512 outputs
  - hidden2               (None, 512)             produces     512 outputs
  - hidden3               (None, 512)             produces     512 outputs
  - output                (None, 3)               produces       3 outputs


-------  ------------  ------------  -----------  -----------  ------
epoch    train loss    valid loss    train/val    valid acc  dur

      1       0.66219       0.60958      1.08630      0.69423  41.58s
      2       0.57866       0.59109      0.97897      0.70242  40.85s
      3       0.55572       0.58235      0.95428      0.70537  37.79s
      4       0.54060       0.57536      0.93958      0.70981  38.03s
      5       0.52828       0.57180      0.92389      0.71234  38.04s
      6       0.51770       0.56730      0.91258      0.71567  37.14s
      7       0.50804       0.56521      0.89886      0.71361  37.25s
      8       0.49913       0.56197      0.88817      0.71487  37.10s
      9       0.49071       0.56017      0.87601      0.71635  35.98s
     10       0.48266       0.55716      0.86627      0.71910  35.56s
     11       0.47496       0.55581      0.85453      0.72058  35.22s
     12       0.46738       0.55352      0.84437      0.72327  35.13s
     13       0.45999       0.55257      0.83245      0.72353  35.03s
     14       0.45277       0.55060      0.82233      0.72400  35.22s
     15       0.44558       0.55000      0.81015      0.72422  38.58s
     16       0.43850       0.54874      0.79911      0.72506  35.52s
     17       0.43147       0.54809      0.78722      0.72527  34.98s
     18       0.42436       0.54800      0.77438      0.72548  34.89s
     19       0.41736       0.54780      0.76188      0.72506  34.50s
     20       0.41039       0.54682      0.75050      0.72459  34.36s
     21       0.40344       0.54638      0.73839      0.72607  35.60s
     22       0.39647       0.54548      0.72682      0.72585  35.24s
     23       0.38953       0.54547      0.71411      0.72712  34.92s
     24       0.38267       0.54578      0.70115      0.72775  34.43s
     25       0.37580       0.54576      0.68858      0.72585  34.84s
     26       0.36889       0.54631      0.67525      0.72649  35.13s
     27       0.36212       0.54652      0.66258      0.72543  34.67s
     28       0.35543       0.54745      0.64925      0.72522  34.86s
     29       0.34888       0.54842      0.63615      0.72712  36.39s
     30       0.34235       0.54962      0.62289      0.72670  35.12s

* f1 score: 0.724699683878
*             precision    recall  f1-score   support

          0       0.73      0.81      0.77      3271
          1       0.71      0.62      0.66      2531

* avg / total       0.72      0.73      0.72      5802

* We've selected the Neural Network as our MVP Portrait Classification.

-------  ------------  ------------  -----------  -----------  ------

### Artist Classification (ArtistLearning.py & neuralNet.py)
* We chose the following three painters for our classification dataset: Cezanne, Van Gogh, and Joseph Mallord Turner.

* Before building our supervised learning models, we preprocessed our images. We resized our images to 480 x 480, extracted 30 random 80 x 96 patches from the 480 x 480 images, calculated the dominant colors for each patch, grayscaled them, applied total variation denoising, and implemented canny edge detection.

* To test whether we can build an painting image detection model, we decided to experiment with Random Forest, Gradient Boosting, Support Vector Machine, K Nearest Neighbor Classifiers, and a Neural Network to determine the painting's artist. 

* We obtained the following F1 scores by cross-validation where K = 8:
    - RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
        - F1 CV Score: 0.61 (+/- 0.01)
    - GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
              max_depth=3, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              random_state=None, subsample=1.0, verbose=0,
              warm_start=False)
        - F1 CV Score: 0.66 (+/- 0.01)
    - KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=5, p=2, weights='uniform')
        - F1 CV Score: 0.50 (+/- 0.01)

* For our deep learning, we only had to resize our images to 360 x 360, extracted 30 random 80 x 96 patches from the 360 x 360 images for our data preprocessing. We built a Neural Network with three hidden layers which each have 512 outputs. Here are the results:

  - input                 (None, 23040)           produces   23040 outputs
  - hidden1               (None, 512)             produces     512 outputs
  - hidden2               (None, 512)             produces     512 outputs
  - hidden3               (None, 512)             produces     512 outputs
  - output                (None, 3)               produces       3 outputs

-------  ------------  ------------  -----------  -----------  ------
epoch    train loss    valid loss    train/val    valid acc  

      1       0.79505       0.73670      1.07921      0.68776  24.97s
      2       0.69658       0.69073      1.00847      0.70283  26.31s
      3       0.65909       0.66561      0.99021      0.70869  23.67s
      4       0.63393       0.65166      0.97279      0.71409  23.09s
      5       0.61424       0.64091      0.95838      0.71914  22.32s
      6       0.59769       0.63162      0.94627      0.72333  21.50s
      7       0.58299       0.62479      0.93310      0.72472  21.74s
      8       0.56971       0.61927      0.91997      0.72724  21.17s
      9       0.55756       0.61429      0.90765      0.72919  21.62s
     10       0.54600       0.61034      0.89457      0.73062  21.45s
     11       0.53494       0.60755      0.88049      0.73201  21.39s
     12       0.52437       0.60411      0.86800      0.73396  21.53s
     13       0.51411       0.60173      0.85440      0.73428  21.54s
     14       0.50421       0.59906      0.84166      0.73651  21.58s
     15       0.49441       0.59702      0.82814      0.73679  21.63s
     16       0.48495       0.59522      0.81475      0.73958  21.43s
     17       0.47553       0.59355      0.80116      0.73986  21.38s
     18       0.46630       0.59174      0.78802      0.74101  21.18s
     19       0.45712       0.59090      0.77360      0.73933  21.16s
     20       0.44814       0.58951      0.76019      0.74073  23.05s
     21       0.43917       0.58911      0.74548      0.74101  21.28s
     22       0.43029       0.58771      0.73215      0.74296  21.92s
     23       0.42143       0.58732      0.71755      0.74491  22.25s
     24       0.41273       0.58664      0.70355      0.74603  21.59s
     25       0.40407       0.58589      0.68967      0.74770  21.46s
     26       0.39540       0.58575      0.67503      0.74770  21.59s
     27       0.38687       0.58563      0.66061      0.74687  21.44s
     28       0.37832       0.58602      0.64557      0.74770  21.27s
     29       0.36994       0.58642      0.63084      0.74882  21.26s
     30       0.36158       0.58660      0.61640      0.74966  21.43s


* f1 score: 0.70944276481
*             precision    recall  f1-score   support

          0       0.41      0.33      0.37       635
          1       0.77      0.78      0.78      2534
          2       0.72      0.78      0.75      1235

* avg / total       0.71      0.72      0.71      4404

* We've selected the Neural Network as our MVP Multi-Artist Classification.

-------  ------------  ------------  -----------  -----------  ------

### Web App 

#### mvpNeuralNet.py

* Defined our Artist Classifier and Genre Classifier Classes.
  - Each Class unpickles a trained neural network.


#### uploadImagePipeline.py
* Performs preprocessing on the uploaded image in the web app and classifies the image according to
whichever classifier we choose. 
  - Image is resized on 360 x 360 pixels and broken up into 30 80 x 96 patches
  - We then predict the classifcations of the patches and perform a majority vote to determine the classification of the full image.

#### app.py

* app.py conatins the whole Flask webapp that allows users to upload an image and obtain a classification of the image's artist/genre.

-------  ------------  ------------  -----------  -----------  ------

### Next Steps to Improve Classifier

* Dig deeper into the rabbit hole of deep learning by implementing a convultional neural network (). We wrote a convultional neural network which is a knock Alex Krizhevsky's ImageNet NN Classifier.

-------  ------------  ------------  -----------  -----------  ------