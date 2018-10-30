from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



"""
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, documents):
        return [self.wnl.lemmatize(word) for word in word_tokenize(documents)]

text = ["The quick brown fox jumped over the lazy dog.",
		"The dog.",
		"The foxes"]
tf_vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(),
                                strip_accents = 'unicode', # works
                                stop_words = 'english', # works
                                lowercase = True, # works
                                )

tf_vectorizer.fit(text)
print("vectorizer.vocabulary_",tf_vectorizer.vocabulary_)
print(tf_vectorizer.idf_)
# encode document
vector = tf_vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())
"""


def nltkTokenizer(collection):
    stemmedTokens = []
    for document in collection:
        tokens = word_tokenize(document)
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # filter out stop words

        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        #print(words[:100])
        # stemming of words
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in words]
        stemmedTokens.append(stemmed)
        #print(stemmed[:100])
    return stemmedTokens
#print(nltkTokenizer(collection))

