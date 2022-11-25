import spacy
import sys
import os
import re
from string import punctuation, whitespace
import pandas as pd
from spacy import displacy
from phrase_extraction import *



def extract_document(folder_name, doc_name):
  # read all .txt files from the input folder
  input_path = 'input/' + folder_name
  documents = read_documents(input_path)
  result_df = pd.DataFrame()
  # extract each sentence and save in results folder
  for file in documents:
    sentence = documents[file]
    doc = nlp(sentence)
    all_spans = doc.spans['sc']
    spans_dict = dict()
    for label, spans in groupby(all_spans, lambda span: span.label_):
        # merge all spans of one category into one string
        merged = ' '.join([span.text.strip(punctuation + whitespace) for span in spans])
        spans_dict[label] = merged
    # output df with extracted spans
    df = pd.DataFrame(
          {
              "EXTRACTED": [sentence] + [spans_dict.get(category, '') for category in categories],
          },
          index = ['SENTENCE'] + categories
      )
    # transpose, add negation column
    df_out = df.transpose()
    df_out["{}_no_neg".format(doc_name)] = df_out.apply(lambda row : get_number_of_negations_in_sentence(row['VERB']), axis = 1)
    result_df = pd.concat([result_df, df_out])
  # change names of dfs
  result_df.columns = result_df.columns.str.replace('SENTENCE', '{}_original_sentence'.format(doc_name))
  result_df.columns = result_df.columns.str.replace('SUBJECT', '{}_sub'.format(doc_name))
  result_df.columns = result_df.columns.str.replace('VERB', '{}_verb'.format(doc_name))
  result_df.columns = result_df.columns.str.replace('TIME', '{}_time'.format(doc_name))
  result_df.columns = result_df.columns.str.replace('CONDITION', '{}_cond'.format(doc_name))
  result_df.columns = result_df.columns.str.replace('OBJECT', '{}_obj'.format(doc_name))
  out_name = f'{folder_name}.xlsx'
  out_path = 'result/' + out_name  
  pd.DataFrame(result_df).to_excel(out_path) 


def read_documents(directory): 
  '''reads in txts of regulatory and realization documents
  Input: multiple .txt files (each a sentence)
  Output: dictionary with file name as key and its content as value'''
  doc_dict = dict()
  files = os.listdir(directory)
  try:
    for fi in files:
        if fi.endswith('.txt'):
          with open(directory+'/'+fi,'r') as f:
              doc_dict[re.sub('\.txt', '', fi)] = f.read()
  except FileNotFoundError:
    print("Wrong file or file path to dir.")
    quit()
  return doc_dict


def get_number_of_negations_in_sentence(text):
    '''extracts the number of explicit negations from a sentence'''
    no_negations = 0
    doc = nlp(text)
    for token in doc:   
        if (token.dep_ == 'neg'):
            no_negations =+ 1
    return no_negations


if __name__ == '__main__':
  # load the previously saved model:
  if len(sys.argv) != 2:
    print("Wrong number of arguments. Please specify the path to spacy model that will extract the spans.")
    quit()
  nlp = spacy.load(sys.argv[1])

  # extract each document from the input folder and save it as html in result folder
  extract_document('realization_document', 'rea')
  extract_document('regulatory_document', 'reg')
