import numpy as np
import spacy
import pandas as pd
from gensim.models import KeyedVectors
#load spacy model
#Need to install spacy version >2.1
# and install vi_spacy
# using: https://github.com/trungtv/vi_spacy
# nlp = spacy.load('vi_spacy_model')
# word2vecmodel_link = "models/wiki.vi.model.bin"
# word2vec = KeyedVectors.load_word2vec_format(fname=word2vecmodel_link,binary=True,unicode_errors='strict')
class ChitChat:
    def __init__(self, datafile):
        #read data file
        df = pd.read_csv(datafile, header=None, names=['intent', 'q', 'noisyq', 'a'])
        qa = df[['q', 'a']].copy()
        # Remove duplicate questions
        qa.drop_duplicates(subset='q', keep='first', inplace=True)
        # print(qa.shape)
        qa_list = qa[['q', 'a']].values
        # print(qa_list)
        # make vectors as spacy.token.doc.Doc type
        targets = [
            q for q, a in qa_list
        ]
        nlp = spacy.load('vi_spacy_model')
        word2vecmodel_link = "models/wiki.vi.model.bin"
        self.word2vec = KeyedVectors.load_word2vec_format(fname=word2vecmodel_link,binary=True,unicode_errors='strict')
        self.targets = []
        self.targets.append(targets)
        self.qa_list = []
        self.qa_list.append(qa_list)
    def retrieve_answer(self, question,ids):
        '''
        Question:: String
        '''
        distances = [self.word2vec.wmdistance(question.split(" "),t.split(" ")) for t in self.targets[ids]]
        id=np.argmin(distances)
        print(distances[id])
        print(id)
        return self.qa_list[ids][id]

    def add_more_data(self,datafile):
        df = pd.read_csv(datafile, header=None, names=['intent', 'q', 'n', 'a'])
        qa = df[['q', 'a']].copy()
        qa_list = qa[['q', 'a']].values
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
    