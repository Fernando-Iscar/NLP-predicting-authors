
# ! pip install TextBlob

import json
import pandas as pd
import numpy as np
import string
import time

# Text polarity
from textblob import TextBlob

# For model-building
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer,HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

# SVD
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.preprocessing import StandardScaler

# Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# For text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


# Get the data

# First, we are going to get the data from the json. Then we are going to process it. And then use ML.

# Read the json file
trainSet = pd.read_json("./train.json")
testSet = pd.read_json("./test.json")
trainSet.head(3)

# We are going to remove the papedId and authorName because it won't tell us nothing about the author. (We already have authorId to identify the author, so let's remove authorName too).

trainSet = trainSet.drop('authorName',axis = 1)
trainSet = trainSet.drop('paperId',axis = 1)
testSet = testSet.drop('paperId',axis = 1)

# Now, let's have a look to what we have:

# We take an example: The amount of data for authorId 3188285

print("Amount of data:",len(trainSet.loc[trainSet["authorId"]==3188285]))
# First instance:
print(trainSet.loc[trainSet["authorId"]==3188285].iloc[1])
# All the venues from this author
print(trainSet.loc[trainSet["authorId"]==3188285,["venue"]])

# So for every author we must somehow encode the title and the abstract. 
# Furthermore, we might deduce from the previous example that author likes to write about a topic related with venue "CLPsych". 
# Thus, we can consider splitting venues.

# Also, we can merge title and abstract. The "Type of writing" is the same.

# Merge title and abstract
trainSet["text"] = trainSet["title"] + '. '+ trainSet["abstract"]
trainSet = trainSet.drop('abstract',axis = 1)
trainSet = trainSet.drop('title',axis = 1)
# Same in test dataset
testSet["text"] = testSet["title"] + '. '+ testSet["abstract"]
testSet = testSet.drop('abstract',axis = 1)
testSet = testSet.drop('title',axis = 1)


# Split venues
def splitVenues(dataSet):
  dataSet["venue"] = (dataSet["venue"].str.replace("2\d{3}","", regex = True)).str.strip();
  arr = []
  for i in dataSet["venue"]:
      if i.find('@') == -1:
          arr.append(["",i])
      else:
          arr.append(i.split("@"))
        
  dataSet = pd.concat([dataSet, pd.DataFrame(arr,columns = ["campus","uni"])],axis = 1).drop("venue",axis = 1)
  return dataSet

trainSet = splitVenues(trainSet)
testSet = splitVenues(testSet)
trainSet.head(3)

# Now let's keep the objective that we are trying to predict aside.

# We are going to try and predict the authorId
goal = trainSet["authorId"]

#Lets remove also the id
trainSet = trainSet.drop('authorId',axis = 1)

# Finally, before we start processing text, we might want to split the years also:

# Fine, so to start processing the text we are going to obtain the lemmas, for example, having -> have.

# Credits https://medium.com/analytics-vidhya/nlp-tutorial-for-text-classification-in-python-8f19cd17b49e

# Convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text
 
# STOPWORD REMOVAL(deleting neutral words such as and)
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

# LEMMATIZATION (to obtain words lemmas)

# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

# Here we just preprocess the strings, removed the stopwords and lemmatized them. You can find more about this in:
# 
# *   https://www.kaggle.com/code/andreshg/nlp-glove-bert-tf-idf-lstm-explained
# *   https://medium.com/analytics-vidhya/nlp-tutorial-for-text-classification-in-python-8f19cd17b49e
# 

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))
trainSet['clean_text'] = trainSet['text'].apply(lambda x: finalpreprocess(x))
trainSet = trainSet.drop('text', axis = 1)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))
testSet['clean_text'] = testSet['text'].apply(lambda x: finalpreprocess(x))
testSet = testSet.drop('text', axis = 1)

# We are going to save this file, because getting the lemmas is a heavy process.

testSet.head()

# Let's count the words in every text, unique words, characters, stop_words, average word length
# punctiation, text polarity, subjectivity...

def useful_text_features(og_train):
    df = pd.DataFrame()
    df["num_words"] = (og_train["title"]+". "+og_train["abstract"]).apply(lambda x: len(str(x).split()))
    df["num_uniq_words"] = (og_train["title"]+". "+og_train["abstract"]).apply(lambda x: len(set(str(x).split())))
    df["chars"] = (og_train["title"]+". "+og_train["abstract"]).apply(lambda x: len(str(x)))
    df["stop_words"] = (og_train["title"]+". "+og_train["abstract"]).apply(
      lambda x: len([w for w in str(x).lower().split() if w in stopwords.words("english")]))

    df["num_punc"] = (og_train["title"]+". "+og_train["abstract"]).apply(
      lambda x: len([w for w in str(x)  if w in list(string.punctuation)]))

    df["avg_word_len"] = (og_train["title"]+". "+og_train["abstract"]).apply(
      lambda x: np.mean([len(w) for w in str(x).split()]))

    df["text_polarity"] = (og_train["title"]+". "+og_train["abstract"]).apply(
      lambda x: TextBlob(x).sentiment[0])
    df["text_subj"] = (og_train["title"]+". "+og_train["abstract"]).apply(
      lambda x: TextBlob(x).sentiment[1])
    return df

og_train = pd.read_json("./train.json")
og_test = pd.read_json("./test.json")

trainSet[["num_words", "num_uniq_words", "chars", "stop_words", "num_punc", "avg_word_len","text_polarity","text_subj"]] = useful_text_features(og_train)[["num_words", "num_uniq_words", "chars", "stop_words", "num_punc", "avg_word_len","text_polarity","text_subj"]]
   
testSet[["num_words", "num_uniq_words", "chars", "stop_words", "num_punc", "avg_word_len","text_polarity","text_subj"]] = useful_text_features(og_test)[["num_words", "num_uniq_words", "chars", "stop_words", "num_punc", "avg_word_len","text_polarity","text_subj"]]      


# We're going to save the lemmatized document, to run code faster next time
trainSet.to_json('data_lemmatized.json')
testSet.to_json('test_data_lemmatized.json')

# Focus on the words:

# So after we removed the useless words, our approach is to get the common words 
# among the different author publications and also the test dataSet.
# Hopefully this will help sweeping up the useless words 
# that don't really gather the author's speech intention.

trainSet = pd.read_json("./data_lemmatized.json") #c
og_train = pd.read_json("./train.json") #c
testSet = pd.read_json("./test_data_lemmatized.json") #c
trainSet = pd.concat([trainSet, og_train[["authorId","paperId"]]], axis = 1)

# We tokenize words:

test_word_set = set()
for i in testSet["clean_text"]:
    test_word_set.update(set(nltk.word_tokenize(i) ))
1692242

new_text = []

for index, row in trainSet.iterrows():
    other_val = trainSet[lambda x: (x["authorId"] == row["authorId"]) & (x["paperId"] != row["paperId"])]
    if(other_val.shape == 0):
        new_text.append(row["clean_text"])
        continue
    tokens = nltk.word_tokenize(row["clean_text"])
    valores = list()
    for token in tokens:
        if(token in test_word_set):
            valores.append(token)
        else:
            for index, value in other_val["clean_text"].items():
                if value.find(token)>=0:
                    valores.append(token)
    new_text.append(" ".join(valores))
    #print(valores,row["clean_text"])

trainSet["clean_text"] = new_text
nuevo_text = None

trainSet.to_json("common_words_dataset.json")

# Analyzing the text(encoding all the text features for computer to understand):

# We are going to start back from the saved data
trainSet = pd.read_json("./common_words_dataset.json")
# og_train = pd.read_json("./Desktop/EjercicioML/train.json")
# testSet = pd.read_json("./Desktop/EjercicioML/test_data_lemmatized.json")
goal = trainSet["authorId"]
trainSet.head(3)

# dataSet = dataSet.drop("campus",axis = 1)
trainSet = trainSet.drop("authorId",axis = 1 )
trainSet = trainSet.drop("paperId",axis = 1 )

# We used these "few" lines just to work with less data willing to improve speed:
data_new = trainSet
new_goal = goal
trainSet = None
goal = None

# This counts the number of nouns, pronouns, verbs, adjectives...
def pos_count(sent):
    nn_count = 0   #Noun
    pr_count = 0   #Pronoun
    vb_count = 0   #Verb
    jj_count = 0   #Adjective
    uh_count = 0   #Interjection
    cd_count = 0   #Numerics
    
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    for token in sent:
        if token[1] in ['NN','NNP','NNS']:
                    nn_count += 1
        if token[1] in ['PRP','PRP$']:
                    pr_count += 1
        if token[1] in ['VB','VBD','VBG','VBN','VBP','VBZ']:
                    vb_count += 1
        if token[1] in ['JJ','JJR','JJS']:
                    jj_count += 1
        if token[1] in ['UH']:
                    uh_count += 1
        if token[1] in ['CD']:
                    cd_count += 1
    
    return pd.Series([nn_count, pr_count, vb_count, jj_count, uh_count, cd_count])


data_new[["nn_count", "pr_count", "vb_count", "jj_count", "uh_count", "cd_count"]] = data_new["clean_text"].apply(pos_count)

# Credits:
# *   https://betterprogramming.pub/beginners-to-advanced-feature-engineering-from-text-data-c228047a4813

# We need to transform campus, uni to float
enc = LabelEncoder()
enc.fit(list(data_new["campus"])+list(data_new["uni"]))
data_new["campus"] = enc.transform(data_new["campus"])
data_new["uni"] = enc.transform(data_new["uni"])

# We are going to count the number of appearances of each word
# Tfidf doesn't improve accuracy, but maybe we should try others
cv = CountVectorizer()
X = cv.fit_transform(data_new["clean_text"])
X = pd.DataFrame(X.toarray())

# X.columns = cv.get_feature_names_out()
X.index = data_new.index
data_new = pd.concat([data_new,X],axis = 1).drop("clean_text",axis = 1)
X = None
data_new.head(2)

x_train, x_val, y_train, y_val = train_test_split(data_new, new_goal, test_size=0.1, random_state = 42)
x_train.head()

# Freeing Memory
data_new = None
new_goal = None

"""clf = RandomForestClassifier( verbose=1,n_jobs = 4)
clf.fit(x_train, y_train)
print("Accuracy of Model :", clf.score(x_val, y_val))

# We are going to optimize the RFC:

# Number of trees in random forest (trying to implement hyperparameter tunning for random forest )
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1500, num = 7)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(50, 110, num = 4)]
#max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1,3,5]
# Method of selecting samples for training each tree
bootstrap = [True]
# Create the random grid
random_grid = {'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier(n_jobs = 3)
# Random search of parameters, using 3 fold cross validation, 
# Search across 100 different combinations, and use all available cores
rf_random = GridSearchCV(estimator = rf, param_grid = random_grid,cv=2, verbose=3, n_jobs = 1)
# Fit the random search model
rf_random.fit(x_train.values, y_train.values) """

"""print("\n The best estimator across ALL searched params:\n", rf_random.best_estimator_)
print("\n The best score across ALL searched params:\n", rf_random.best_score_)
print("\n The best parameters across ALL searched params:\n", rf_random.best_params_)"""

mult=MultinomialNB()
mult.fit(x_train, y_train)
alp ={'alpha':(1,0.1,0.01,0.05,0.001,0.005,0.0001,0.0005)}
#hyperpara
grid_mult=GridSearchCV(mult, alp,n_jobs=-1)
grid_mult.fit(x_train, y_train)

print(grid_mult.score(x_val, y_val))

# Final train + writing the solution

# We start off with the processed data from the first section:

#Let's train the model again
trainSet = pd.read_json("./common_words_dataset.json")
testSet = pd.read_json("./test_data_lemmatized.json")

trainSet = trainSet.drop("paperId",axis=1)
#testSet = splitVenues(testSet)
y = trainSet["authorId"]
trainSet = trainSet.drop("authorId",axis=1)
trainSet.head(1)

trainSet[["nn_count", "pr_count", "vb_count", "jj_count", "uh_count", "cd_count"]] = trainSet["clean_text"].apply(pos_count)
testSet[["nn_count", "pr_count", "vb_count", "jj_count", "uh_count", "cd_count"]] = testSet["clean_text"].apply(pos_count)

#transform it to floats
enc = LabelEncoder()
enc.fit(list(trainSet["campus"])+list(trainSet["uni"])+list(testSet["campus"])+list(testSet["uni"]))
trainSet["campus"] = enc.transform(trainSet["campus"])
trainSet["uni"] = enc.transform(trainSet["uni"])
testSet["campus"] = enc.transform(testSet["campus"])
testSet["uni"] = enc.transform(testSet["uni"])

# We are going to count the number of appearances of each word
# Tfidf doesn't improve accuracy, but maybe we should try others
cv = CountVectorizer()
# We first fit all values
cv.fit(list(trainSet["clean_text"])+list(testSet["clean_text"]))
# And then transform just the data/test
X = cv.transform(trainSet["clean_text"])
X = pd.DataFrame(X.toarray())
# X.columns = cv.get_feature_names_out()
X.index = trainSet.index
trainSet = pd.concat([trainSet,X],axis = 1).drop("clean_text",axis = 1)

X = cv.transform(testSet["clean_text"])
X = pd.DataFrame(X.toarray())
# X.columns = cv.get_feature_names_out()
X.index = testSet.index
testSet = pd.concat([testSet,X],axis = 1).drop("clean_text",axis = 1)
X = None
trainSet.head(2)

#selected Multinomial 
mlt=MultinomialNB()
mlt.fit(trainSet, y)
alp ={'alpha':(1,0.1,0.01,0.05,0.001,0.005,0.0001,0.0005)}
grid_mlt=GridSearchCV(mlt, alp)
grid_mlt.fit(trainSet, y)

# Predictions (generating and writing them in the right format)
predictions = grid_mlt.predict(testSet)
str(predictions)

predictions = pd.DataFrame([str(x) for x in predictions], columns = ["authorId"])

pd.concat([og_test["paperId"], predictions ],axis = 1).to_json("pruebadoc.json",orient='records', indent=4 )


