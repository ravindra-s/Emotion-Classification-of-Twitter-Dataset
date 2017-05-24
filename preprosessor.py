import logging
import pandas as pd
import re
from nltk.corpus import stopwords
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

def clean_text(raw_text, keep_emoji):

	if keep_emoji == True:
		emoji_pat = '[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]'

		reg = re.compile(r"({})|[^a-zA-Z]".format(emoji_pat)) # line a
		letters_and_emoji = reg.sub(lambda x: " {} ".format(x.group(1)) if x.group(1) else " ", raw_text)
	else:
		# keep only words
		letters_and_emoji = re.sub("[^a-zA-Z]", " ", raw_text)

	# convert to lower case and split 
	words = letters_and_emoji.lower().split()

	# remove stopwords
	stopword_set = set(stopwords.words("english"))
	meaningful_words = [w for w in words if w not in stopword_set]

	cleaned_word_list = " ".join(meaningful_words)

	return cleaned_word_list


def preprocess(dataset, keep_emoji):
	
	logger.debug("Beginning processing of tweets")

	tweets_df = pd.read_csv(dataset,delimiter='|',header=None)
	num_tweets = tweets_df.shape[0]
	cleaned_tweets = []
	cleaned_tweets_labels = []

	logger.info("Total tweets before beginning the cleaning process: " + str(num_tweets))
	
	empty_tweet_count = 0
	cleaned_tweets_count = 0

	for i in range(num_tweets):
		old_len = len(tweets_df.iloc[i][1])
		cleaned_tweet = clean_text(tweets_df.iloc[i][1], keep_emoji)
		
		if len(cleaned_tweet) == 0:
			empty_tweet_count += 1
		else:
			cleaned_tweets_count +=1 
			cleaned_tweets.append(cleaned_tweet)
			cleaned_tweets_labels.append(tweets_df.iloc[i][0])

		if(i % 20000 == 0):
			logger.info(str(i) + " tweets processed")
		
	logger.debug("Total zero length tweets after cleaning " + str(empty_tweet_count))
	logger.debug("Total tweets after cleaning process is finished: " + str(cleaned_tweets_count))
	logger.debug("Finished processing of tweets")


	return cleaned_tweets, cleaned_tweets_labels

def preprocess_dataset(dataset_file, keep_emoji = False):
	logger.info("Begin preprocessing of :" + str(dataset_file))
	logger.info("Whether to use emoji as feature? : " + str(keep_emoji))

	cleaned_tweets, cleaned_tweets_labels = None, None

	if keep_emoji == False:
		words_pickle = "cleaned_dataset_without_emoji.pkl"
		labels_pickle = "cleaned_labels_without_emoji.pkl"

	else: 
		words_pickle = "cleaned_dataset_with_emoji.pkl"
		labels_pickle = "cleaned_labels_with_emoji.pkl"

	if Path(words_pickle).is_file() == False:
		# Option 1: Load CSV, clean it and dump it in a pickle
		logger.info("No dataset pickle file found. Generating it now.")
		logger.info("Loading dataset from csv and cleaning it : " + dataset_file)
		cleaned_tweets, cleaned_tweets_labels = preprocess(dataset_file, keep_emoji)
		words_pickle = open(words_pickle,'wb')
		labels_pickle = open(labels_pickle,'wb')
		pickle.dump(cleaned_tweets, words_pickle)
		pickle.dump(cleaned_tweets_labels, labels_pickle)
	else:
		# Option 2: Load an existing pickle
		logger.info("Dataset and label pickle files found. Loading them now.")
		logger.info("Loading cleaned dataset from pickle file : " + words_pickle)
		cleaned_tweets = pickle.load(open(words_pickle,'rb'))
		logger.info("Loading corresponding labels from pickle file : " + labels_pickle)
		cleaned_tweets_labels = pickle.load(open(labels_pickle,'rb'))
	
	logger.info("Finished preprocessing of : " + str(dataset_file))
	
	return cleaned_tweets, cleaned_tweets_labels