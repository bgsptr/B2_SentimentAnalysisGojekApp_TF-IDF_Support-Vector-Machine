import streamlit as st
import pickle
import time 
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import pandas as pd
from io import StringIO

###############################################################################
######################### IMPORT GBC MODEL ####################################
###############################################################################
modelsvm = pickle.load(open("gojek_svm.pkl", "rb"))

###############################################################################
######################### FUNCTION DECLARATION ################################
###############################################################################

def casefolding(teks):
    # transform text into lowercase
    teks = teks.lower()
    
    # remove the punctuation and special characters 
    teks = re.sub(r'[^\w\s]', ' ', teks)
    
    # remove digits
    teks = re.sub(r'[\d+]', '', teks)
    
    # remove greeks
    teks = (re.sub('(?![ -~]).', '', teks))
    
    return teks
    
def tokenization(teks):
    # split the text into a series of tokens
    teks = word_tokenize(teks)
    
    return teks

def normalisasi(teks):
    kamusSlang = eval(open("kamus.txt").read())
    pattern = re.compile(r'\b( ' + '|'.join (kamusSlang.keys())+r')\b')
    content = []
    for kata in teks:
        filterSlang = pattern.sub(lambda x: kamusSlang[x.group()],kata)
        content.append(filterSlang.lower())
    teks = content
    return teks

#def removedStopwords(text):
#    
#    # initialize an empty list of tokens with no stopwords
#    tokens_with_no_stopwords = list()
#    
#    # extract tokens not found in a list of stopwords 
#    for word in text:
#        if word not in stopwords.words('english'):
#            tokens_with_no_stopwords.append(word)
#    
#    return tokens_with_no_stopwords

#def removedStopwords(teks):
#    tokens_with_no_stopwords = list()
#
#    for word in teks:
#        if word not in stopwords_indonesia:
#            tokens_with_no_stopwords.append(word)
#
#    return tokens_with_no_stopwords
#penghapusan stopwords
stopwords_indonesia = set(stopwords.words('indonesian'))

def removedStopwords(teks):
    tokens_with_no_stopwords = list()

    for word in teks:
        if word not in stopwords_indonesia:
            tokens_with_no_stopwords.append(word)

    return tokens_with_no_stopwords
 
def removed_words_less_than_4_characters(teks):
    # initialize an empty list for tokens with more than 4 characters
    tokens_with_more_than_4_characters = list()
    
    # extract tokens with more than 4 characters
    for word in teks:
        if len(word) >= 4:
            tokens_with_more_than_4_characters.append(word)
            
    return tokens_with_more_than_4_characters

    
#def wordsLemmatization(text):
#    # define object for lemmatizer 
#    lemmatizer = WordNetLemmatizer()
#    
#    # initialize an empty list for lemmatized words
#    wordsLemmatize = list()
#    
#    # lemmatize the words
#    for word in text:
#        wordsLemmatize.append(lemmatizer.lemmatize(word))
#    
#    return wordsLemmatize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(teks):
    return [stemmer.stem(word) for word in teks]

def sentence_reconstruction(teks):
    # initialize an emppty string 
    sentence_reconstruction = ""
    
    # combine each token to form a string 
    for word in teks:
        sentence_reconstruction = sentence_reconstruction + word + " "
    return sentence_reconstruction 

def sentiment_prediction(teks):
    predict_sentiment = modelsvm.predict([teks])
    return predict_sentiment

def main():
    
    ###############################################################################
    ################################### MAIN  #####################################
    ###############################################################################
    st.title("Analisis Sentimen Gojek")
    st.markdown("**Mengimplementasi SVM dan TF-IDF untuk klasifikasi sentimen**")
    st.image("logo_gojek.png")
    # instructions to the user
    st.write("**Instructions:**")
    st.info("Inputkan csv (kolom teks bernama content) yang akan dilakukan analisis sentimen lalu lakukanlah pencet tombol **SUBMIT** . ")
    # prompt the user to enter a text 
#    teks = st.text_area("Masukkan teks: ", value = "", placeholder = "Sisipkan teks di sini.")
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        teks = pd.read_csv(uploaded_file).head(20)
        df=teks['content']
        st.dataframe(df)

    # prompt the user to click on the button to submit the text for analysis 
    submit_button = st.button("SUBMIT")
    
    if submit_button: 
      # display the raw text      
      with st.spinner("TEXT CLEANING IN PROGRESS!"):
          time.sleep(5)
      # call the function to normalized the text
      df=df.apply(casefolding)
      
      # call the function to tokenized the text
      df=df.apply(tokenization)

      # call the function to tokenized the text
      df=df.apply(normalisasi)
      
      # call the function to removed stopwords
      df=df.apply(removedStopwords)
      
      # call the function to removed words less than 4 characters
      df=df.apply(removed_words_less_than_4_characters)
      
      # call the function to lemmatized the text
      df=df.apply(stemming)
      
      # call the function to rejoined the text after cleaning
      df=df.apply(sentence_reconstruction)
      
      # display the cleaned text
      st.write("**Cleaned Text:** ")
      st.dataframe(df)
      
      with st.spinner("SENTIMENT PREDICTION IN PROGRESS! "):
          time.sleep(5)
      
      # call the function to predict the sentiment text 
      teks['sentimen'] = df.apply(sentiment_prediction)
      teks['content'] = df
      st.markdown("**Prediction Outcome:** ")
      st.dataframe(teks[['content','sentimen']])


if __name__ == '__main__':
    main()
