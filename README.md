# Spam Review Detector 
Machine learning model made using a Jupyter notebook with Numpy, Pandas, and Scikit-Learn that differentiates between real and fake hotel 
reviews (specifically in the Chicago area) with approximately 90% accuracy on average using the given data.

## The Data
The data used to train and test the model was taken from [here](https://myleott.com/op-spam.html). It contains 1600 reviews of hotels in Chicago, of which: 400 are real reviews with positive polarity; 400 are real reviews with negative polarity; 400 are fake reviews with positive polarity; and 400 are fake reviews with negative polarity. I compiled all 1600 reviews into this [.csv file](https://github.com/JakeAMarq/Spam-Review-Detector/blob/master/spamReviewData.csv) that has columns for polarity (0 for positive review, 1 for negative review), spam (0 for real review, 1 for fake review), and text (the review itself).

## The Model
The most important part of the code is:
````python
clf = Pipeline([('vect', TfidfVectorizer(min_df= 3, sublinear_tf=True, norm='l2', ngram_range=(1, 3))),
                        ('chi',  SelectKBest(chi2, k=1200)),
                        ('clf', SVC())])

model = clf.fit(X_train, y_train)
````
Here, a pipeline is created that converts the reviews into a matrix of TFIDF features, trims it down to only the 1200 best features, and uses the result to train a support vector classifier which can then be used to predict if a review is real or fake.
