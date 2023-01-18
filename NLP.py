#NLP Pipeline demonstration for a simple sentence using NLTK Library
# Take an Input String
# Preprocessing Steps
# except BaseException as e: - print(str(e))

# Step 1 - Convert String to Lower Case

input_str = "I Love Python 3.3 Programming; Python is beautiful Language"
input_str = input_str.lower()
original_str = input_str
print("Lower: " + input_str)

# Step 2 - Remove Numbers - Only when needed
from operator import itemgetter
import re

input_str = re.sub(r'\d+', '', input_str)
print("After removing numbers: " + input_str)

# Step 3 - Remove Punctuations
# The following punctuations are removed - [!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]:

import string

punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''

# Removing punctuations in string
# Using loop + punctuation string
for ele in input_str:
    if ele in punc:
        input_str = input_str.replace(ele, " ")

print("After removing Punctuations: " + input_str)

# Step 4 - Remove Whitespaces
input_str = input_str.strip()  # Removes only first and last whitespace
print("After removing Spaces: " + input_str)

import nltk
#nltk.download('all')

# Step 5 - Tokenize the Sentence
from nltk.tokenize import sent_tokenize
print(sent_tokenize(input_str))

# Step 6 - Remove Stop Words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize
print(input_str)

# Perform Word Tokenize
tokens = word_tokenize(input_str)
print("Tokens: " + str(tokens))
result = [i for i in tokens if not i in stop_words]
print("After Word Tokenization: " + str(result))

# Step 7 - Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print("After Lemmatization: " + lemmatizer.lemmatize(input_str))

# Step 8 - Count Word Frequency
tokens = [t for t in input_str.split()]
freq = nltk.FreqDist(tokens)
for key, val in freq.items():
    print(str(key) + ':' + str(val))

print(freq.tabulate())

# Step 9 - Parts of Speech (POS) Tagging
#CC	coordinating conjunction
#CD	cardinal digit
#DT	determiner
#EX	existential there
tokenized = sent_tokenize(input_str)
for i in tokenized:
    # Word tokenizers is used to find the words
    # and punctuation in a string
    wordsList = nltk.word_tokenize(i)

    # removing stop words from wordList
    wordsList = [w for w in wordsList if not w in stop_words]

    #  Using a Tagger. Which is part-of-speech
    # tagger or POS-tagger.
    tagged = nltk.pos_tag(wordsList)
    print(tagged)

# Step 10 - Named Entity Recognition
# Extracting Named Entitites - https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da

ne_tree = nltk.ne_chunk(tagged, binary=False)
print(ne_tree)
