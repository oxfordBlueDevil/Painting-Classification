### Modelling
* To test whether we can build an painting image detection model, we decided to experiment with Random Forest, Gradient Boosting, Support Vector Machine, K Nearest Neighbor Classifiers to determine whether an image is a Portrait painting or a Landscape painting. 

* We obtained the following F1 scores by cross-validation where K = 8:
	- RandomForestClassifier(n_estimators=1000, oob_score=True):
		- F1 Score: 0.65 (+/- 0.03)
	- GradientBoostingClassifier(learning_rate=0.1, n_estimators=100):
		- F1 Score: 0.62 (+/- 0.02)
	- KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=5, p=2, weights='uniform'):
		- F1 Score: 0.59 (+/- 0.00)

* After cross-validation and experimentation, the Random Forest Classifier where number of estimators is 1000 proved to have highest the F1 score of the classifiers we tested. Thus, we've chosen this RandomForestClassifier Model for our baseline model to determine whether whether an image is a Portrait painting or a Landscape painting. For now, we look to take on a much challenging problem.

* The plan for this project is to build an Artist Classification model. So far the performance of our baseline model gives us hope that we can build a multilabeled Classifier to detect which artist made whichever painting.

imp.read(sub_dirs = ['Cezanne', 'VanGogh', 'Durer', 'JosephMallordTurner'])


Binary Artist Classifier

imp.read(sub_dirs = ['Durer', 'Klimt-and-Expressionism'])

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
F1 CV Score: 0.77 (+/- 0.01)

F1 Score: 0.786101370736

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=5, p=2, weights='uniform')
F1 CV Score: 0.74 (+/- 0.01)
F1 Score: 0.743537414966

GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
              max_depth=3, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              random_state=None, subsample=1.0, verbose=0,
              warm_start=False)
F1 CV Score: 0.78 (+/- 0.01)

F1 Score: 0.80437081424

Three Artist Classifier

imp.read(sub_dirs = ['Cezanne', 'VanGogh', 'JosephMallordTurner'])


RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
F1 CV Score: 0.61 (+/- 0.01)

RandomForestClassifier is Fitting
F1 Score: 0.60847852846

GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
              max_depth=3, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              random_state=None, subsample=1.0, verbose=0,
              warm_start=False)
F1 CV Score: 0.66 (+/- 0.01)

GradientBoostingClassifier is Fitting
F1 Score: 0.660903906442

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=5, p=2, weights='uniform')
F1 CV Score: 0.50 (+/- 0.01)

k Nearest Neighbors is Fitting
F1 Score: 0.486659614239

Four Artists

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
F1 CV Score: 0.49 (+/- 0.01)

RandomForestClassifier is Fitting
F1 Score: 0.491433746998

GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
              max_depth=3, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              random_state=None, subsample=1.0, verbose=0,
              warm_start=False)
F1 CV Score: 0.57 (+/- 0.02)

GradientBoostingClassifier is Fitting
F1 Score: 0.569454386649

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=5, p=2, weights='uniform')
F1 CV Score: 0.41 (+/- 0.01)

k Nearest Neighbors is Fitting
F1 Score: 0.417210263545


Three Artist Neural Network

Training Neural Network
  input               (None, 23040)         produces   23040 outputs
  hidden1             (None, 512)           produces     512 outputs
  hidden2             (None, 512)           produces     512 outputs
  hidden3             (None, 512)           produces     512 outputs
  output              (None, 3)             produces       3 outputs
  epoch    train loss    valid loss    train/val    valid acc  dur
-------  ------------  ------------  -----------  -----------  ------
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
     31       0.35329       0.58653      0.60234      0.74854  22.07s
     32       0.34517       0.58712      0.58790      0.74910  21.84s
     33       0.33709       0.58764      0.57364      0.74938  22.31s
     34       0.32933       0.58835      0.55975      0.74857  21.47s
     35       0.32176       0.58939      0.54592      0.74833  21.32s
     36       0.31403       0.58976      0.53248      0.74693  21.35s
     37       0.30660       0.59135      0.51848      0.74777  21.91s
     38       0.29948       0.59205      0.50583      0.74972  21.52s
     39       0.29238       0.59265      0.49334      0.74805  21.33s
     40       0.28551       0.59432      0.48040      0.74833  21.92s
     41       0.27883       0.59500      0.46862      0.74833  25.44s
     42       0.27200       0.59636      0.45610      0.74888  22.17s
     43       0.26564       0.59828      0.44401      0.75000  22.14s
     44       0.25935       0.59984      0.43237      0.74969  21.95s
     45       0.25288       0.60166      0.42031      0.74885  22.25s
     46       0.24684       0.60320      0.40923      0.74746  21.98s
     47       0.24083       0.60551      0.39773      0.74634  21.66s
     48       0.23558       0.60766      0.38768      0.74718  21.77s
     49       0.22962       0.61036      0.37620      0.74522  22.34s
     50       0.22452       0.61273      0.36642      0.74522  21.75s
     51       0.21869       0.61472      0.35575      0.74494  21.12s
     52       0.21183       0.61797      0.34278      0.74603  21.45s
     53       0.20907       0.62092      0.33671      0.74470  21.48s
     54       0.20177       0.62394      0.32338      0.74581  22.03s
     55       0.19959       0.62722      0.31821      0.74554  21.51s
     56       0.19385       0.63063      0.30739      0.74442  22.15s
     57       0.18548       0.63445      0.29234      0.74470  22.34s
     58       0.18661       0.63749      0.29273      0.74609  21.40s
     59       0.18001       0.64032      0.28113      0.74362  22.83s
     60       0.18082       0.64163      0.28181      0.74749  22.58s
     61       0.17394       0.64724      0.26874      0.74526  22.40s
     62       0.16539       0.65131      0.25394      0.74250  22.46s
     63       0.16232       0.65434      0.24807      0.74609  22.14s
     64       0.16652       0.65800      0.25307      0.74696  22.04s
     65       0.16297       0.65917      0.24723      0.74637  21.71s
     66       0.15819       0.66557      0.23767      0.74554  21.76s
     67       0.15689       0.66714      0.23516      0.74578  21.59s
     68       0.14204       0.67212      0.21133      0.74829  21.50s
     69       0.14630       0.67730      0.21600      0.74501  22.01s
     70       0.16209       0.67326      0.24075      0.74749  22.55s
     71       0.13786       0.68050      0.20259      0.74058  21.61s
     72       0.15332       0.68142      0.22499      0.74526  21.43s
     73       0.12866       0.68855      0.18685      0.74414  22.27s
     74       0.12283       0.69035      0.17793      0.74609  22.39s
     75       0.14547       0.69180      0.21028      0.74526  23.46s
     76       0.11796       0.69696      0.16925      0.74609  21.68s
     77       0.11658       0.70156      0.16618      0.74637  21.32s
     78       0.13331       0.70167      0.19000      0.74581  21.61s
     79       0.12853       0.70248      0.18296      0.74637  21.25s
     80       0.13807       0.70084      0.19701      0.74721  21.73s
     81       0.11007       0.71365      0.15423      0.74330  21.00s
     82       0.10690       0.71826      0.14883      0.74470  21.27s
     83       0.10996       0.71972      0.15279      0.74442  23.20s
     84       0.10270       0.72318      0.14201      0.74386  22.76s
     85       0.10489       0.72120      0.14544      0.74526  22.54s
     86       0.10082       0.72418      0.13921      0.74554  22.20s
     87       0.09085       0.72935      0.12456      0.74581  22.64s
     88       0.10588       0.73053      0.14494      0.74498  21.94s
     89       0.08665       0.73628      0.11768      0.74554  22.87s
     90       0.10952       0.73631      0.14874      0.74275  22.45s
     91       0.08515       0.74258      0.11467      0.74554  22.96s
     92       0.10356       0.73998      0.13995      0.74470  22.59s
     93       0.08883       0.74433      0.11935      0.74498  23.09s
     94       0.09507       0.74429      0.12774      0.74330  22.58s
     95       0.07826       0.75547      0.10359      0.74581  22.54s
     96       0.07730       0.76193      0.10145      0.74526  22.76s
     97       0.07433       0.76529      0.09712      0.74526  22.88s
     98       0.07051       0.76985      0.09159      0.74498  22.81s
     99       0.10090       0.76774      0.13142      0.74023  22.40s
    100       0.07682       0.76878      0.09993      0.74526  22.35s

f1 score: 0.70944276481
             precision    recall  f1-score   support

          0       0.41      0.33      0.37       635
          1       0.77      0.78      0.78      2534
          2       0.72      0.78      0.75      1235

avg / total       0.71      0.72      0.71      4404