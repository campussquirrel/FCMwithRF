import nltk
# load data
#filename = 'metamorphosis_clean.txt'
#file = open(filename, 'rt')
#file.read()
#file.close()
# split into words
#from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class NLTKTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.porter = PorterStemmer()
    def __call__(self, documents):
        #stop_words = set(stopwords.words('english'))
        table = str.maketrans('', '', string.punctuation)
        return [self.porter.stem(word).lower().translate(table) for word in word_tokenize(documents) if not word in open('mystopWords.txt').read() and word.isalpha()]
        #return [self.porter.stem(word).lower().translate(table) for word in word_tokenize(documents) if not word in stop_words and not word in open('mystopWords.txt').read() and word.isalpha()]


"""collection = ["The quick brown Fox jumped over the lazy dog.",
		"The lazy dog.",
		"The jumping foxes"]


vectorizer = TfidfVectorizer(tokenizer=NLTKTokenizer(),
                                strip_accents = 'unicode', # works
                                stop_words = 'english', # works
                                lowercase = True, # works
                                )

vectorizer.fit(collection)
print("vectorizer.vocabulary_",vectorizer.vocabulary_)
#print(vectorizer.idf_)

for doc in collection:
    # encode document
    vector = vectorizer.transform([doc])
    # summarize encoded vector
    print(vector.shape)
    print(vector.toarray())
"""


