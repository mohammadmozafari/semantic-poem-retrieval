from django.shortcuts import render
from rest_framework import permissions
from rest_framework.views import APIView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pickle
import numpy as np
from hazm import *
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from rest_framework import permissions
from gensim.models import FastText
import dill
from hazm import *
import pandas as pd
import gensim
import os



class poem_view(APIView):
    # add permission to check if user is authenticated

    def calc_query_LDA_likelihood(self,lda_model,top_stats, query):
        topic_probs = np.full((1, lda_model.num_topics), 0.0001)
        for topics, probs in top_stats:
            topic_probs[0, topics] = probs
        return np.exp(np.sum(np.log(topic_probs @ lda_model.expElogbeta[:, query])))

    def search(self, q):

        module_dir = os.path.dirname(__file__)  # get current directory


        id2word = gensim.corpora.dictionary.Dictionary.load(os.path.join(module_dir, 'results/lda_dict'))
        corpus_infered = gensim.interfaces.TransformedCorpus.load(os.path.join(module_dir, 'results/lda_inferred_corp'))
        lda_model = gensim.models.LdaMulticore.load(os.path.join(module_dir, 'results/lda_model'))

        lemmatizer = Lemmatizer()
        normalizer = Normalizer()
        text = word_tokenize(q)
        text = [normalizer.normalize(w) for w in text]
        text = [lemmatizer.lemmatize(w) for w in text]
        doc = id2word.doc2idx(text)
        doc = list(filter(lambda a: a != -1, doc))
        res = list(map(lambda x: self.calc_query_LDA_likelihood(lda_model, x, doc), corpus_infered))
        res = np.array(res)
        return np.argsort(res)[::-1]




    # 1. List all
    def get(self, request, *args, **kwargs):


        return Response("تست", status=status.HTTP_200_OK)

    # 2. Create
    def post(self, request, *args, **kwargs):


        query = request.data.get('query')
        if query == "":
            query = "شعر"
        type = request.data.get('type')
        module_dir = os.path.dirname(__file__)  # get current directory

        response_poem = "متاسفانه شعری پیدا نشد"
        if type == 'TFIDF' or type == 'tfidf':

            with open(os.path.join(module_dir, 'poems'), 'rb') as f: poems = pickle.load(f)
            with open(os.path.join(module_dir, 'vectorizer_tfidf'), 'rb') as f: vectorizer_tfidf = pickle.load(f)
            with open(os.path.join(module_dir, 'X_toarray'), 'rb') as f: X_toarray = pickle.load(f)
            query_vector = vectorizer_tfidf.transform([query]).toarray()
            dot_product = X_toarray * query_vector
            similarity = np.sum(dot_product, axis=1)

            relevance_order = np.argsort(similarity)[::-1]
            response_poem = poems[relevance_order[0]]


        if type == 'FastText':
            poems = pd.read_csv(os.path.join(module_dir, 'results/corpus.csv'))
            lemmatizer = Lemmatizer()
            normalizer = Normalizer()
            model = FastText.load(os.path.join(module_dir, 'results/fasttext'))
            words_cv, word_tfidf = dill.load(open(os.path.join(module_dir, 'results/cv_tfidf.pkl'), 'rb'))
            doc_rep, words_rep = pickle.load(open(os.path.join(module_dir, 'results/fasttext_doc_word_rep.pkl'), 'rb'))
            query = word_tokenize(query)
            query = [lemmatizer.lemmatize(normalizer.normalize(y)) for y in query]
            query = words_cv.transform([query])
            query = word_tfidf.transform(query)
            query = query @ words_rep
            scores = doc_rep @ query.T
            response_poem = np.argsort(scores[:, 0])[::-1][0]
            response_poem = poems['beyt'][response_poem]


        if type == 'LDA':

            response_poem = self.search(query)[0]
            poem = []
            filepath = os.path.join(module_dir, 'results/mesra.txt')

            with open(filepath, encoding='utf-8', mode='r') as fp:
                line = fp.readline()
                cnt = 1
                box = ''
                while line:
                    poem.append(line.strip())
                    line = fp.readline()
            response_poem = poem[response_poem]


        return Response(response_poem, status=status.HTTP_200_OK)