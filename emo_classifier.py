import logging
import logging.config
from preprosessor import preprocess_dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import OrderedDict

logger = logging.getLogger(__name__)

def main():

	# 0. Logging
	logging.basicConfig(level=logging.DEBUG,
						format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
						datefmt='%m-%d %H:%M',
						filename='emo_classification.log',
						filemode='w')

	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)

	# 1. Preprocessing of dataset 
	dataset_file = "tweet_balanced_labels_dataset.csv"
	
	all_features, all_labels = preprocess_dataset(dataset_file, keep_emoji = False)

	logger.info("Total size of the dataset (num of tweets): " + str(len(all_features)))

	# temporary 
	# all_features = all_features[:500]
	# all_labels = all_labels[:500]
	
	logger.info("Beginning pipeline execution")
	
	estimators_dict = OrderedDict([('MultiNomialNB', MultinomialNB()), \
						('LinearSVC', LinearSVC()), \
						('LogisticRegression', LogisticRegression(n_jobs = -1)), \
						('DecisionTreeClassifier', DecisionTreeClassifier(min_samples_split=2000, \
																			min_samples_leaf=200))])
						
						# ('MLPClassifier',MLPClassifier(max_iter = 40, learning_rate = 'adaptive', \
						# 								warm_start = True, early_stopping = True))])

	transformers_dict = OrderedDict([('tf-idf', TfidfVectorizer(max_features=150000)), \
						  				('bag_of_words', CountVectorizer(ngram_range=(1, 2), \
						  										max_df=0.6,max_features=150000))]) 
	
	steps = []

	f = open('results_summary_f1score.txt','w')
	f.write("A quick look at f1-scores of all the models\n")
	f.write("-"*50)


	for transformer_name, transformer in transformers_dict.items():
		f.write("\nFeature type: " + transformer_name + "\n\n")
		logger.info("*"*100)
		logger.info("For feature transformation using : " + transformer_name)
		steps.append((transformer_name, transformer))

		for estimator_name, estimator in estimators_dict.items():
			
			logger.info("Running the model with : " + estimator_name)
			steps.append((estimator_name, estimator))
			model = Pipeline(steps)
			predicted_labels = cross_val_predict(model, all_features,all_labels, \
												 cv = 5, n_jobs = -1, verbose = 100)
		
			recall = round(recall_score(all_labels, predicted_labels, average = 'weighted'),2)
			precision = round(precision_score(all_labels, predicted_labels, average = 'weighted'),2)
			f1 = round(f1_score(all_labels, predicted_labels, average = 'weighted'),2)
			report = classification_report(all_labels, predicted_labels)
			conf_matrix = confusion_matrix(all_labels, predicted_labels)
		
			logger.info("recall score: " + str(recall))
			logger.info("precision score: " + str(precision))
			logger.info("f1 score: " + str(f1))
			logger.info("confusion matrix: \n" + str(conf_matrix))
			logger.info("classification report: \n" + report)

			logger.info("*"*70)

			f.write(estimator_name + " : " + str(f1) + "\n")

			del steps[1] 	# remove current estimator and make room for the next one

		f.write("-"*50)
		del steps[0]		# remove current transformer and make room for the next one 

	f.close()
	

if __name__ == '__main__':

	main()
