from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import  pos_tag

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.tokenizer= nltk.RegexpTokenizer(r"\w+")

    def __call__(self, doc):
        return [
            self.wnl.lemmatize(
                w.lower(), t[0].lower()) if t[0].lower() in ["a", "v", "n"] else self.wnl.lemmatize(w.lower()) \
            for w, t in pos_tag(
                self.tokenizer.tokenize(doc)
                )
        ]