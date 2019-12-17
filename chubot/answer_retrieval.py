import numpy as np
import spacy
import pandas as pd
from gensim.models import KeyedVectors
#load spacy model
#Need to install spacy version >2.1
# and install vi_spacy
# using: https://github.com/trungtv/vi_spacy


class ChitChat:
    def __init__(self, datafile):
        #read data file
        df = pd.read_csv(datafile, header=None, names=['intent', 'q', 'noisyq', 'a'])
        qa = df[['q', 'a']].copy()
        # Remove duplicate questions
        qa.drop_duplicates(subset='q', keep='first', inplace=True)

        qa_list = qa[['q', 'a']].values

        # make vectors as spacy.token.doc.Doc type
        targets = [
            q for q, a in qa_list
        ]
        self.targets=[]
        self.qa_list=[]
        word2vecmodel_link = "models/wiki.vi.model.bin"
        self.word2vec = KeyedVectors.load_word2vec_format(fname=word2vecmodel_link,binary=True,unicode_errors='strict')
        self.targets.append(targets) 
        self.qa_list.append(qa_list)
        # self.nlp = spacy.load('vi_spacy_model')
    def retrieve_answer(self, question,id_number):
        '''
        Question:: String
        '''
        distances = [self.word2vec.wmdistance(question.split(" "),t.split(" ")) for t in self.targets[id_number]]
        print(self.targets[id_number])
        id=np.argmin(distances)
        return self.qa_list[id_number][id]

    def add_more_data(self,datafile):
        df = pd.read_csv(datafile, header=None, names=['intent', 'q', 'noisyq', 'a'])
        qa = df[['q', 'a']].copy()
        # Remove duplicate questions
        qa.drop_duplicates(subset='q', keep='first', inplace=True)

        qa_list = qa[['q', 'a']].values

        # make vectors as spacy.token.doc.Doc type
        targets = [
            q for q, a in qa_list
        ]
        self.targets.append(targets)
        self.qa_list.append(qa_list)

if __name__ == '__main__':
    chitchat_file = 'data/chitchat.csv'
    chitchat = ChitChat(chitchat_file)
    
    # input_question = 'bạn thích gì nhất?'
    input_question = 'hiệu trưởng trường đại học công nghệ là ai'
    most_similar_question, answer = chitchat.retrieve_answer(input_question)
    
    print("inputquestion: ", input_question)
    print("most similar question found: ", most_similar_question)
    print("answer: ", answer)
    