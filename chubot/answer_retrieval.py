import numpy as np
import spacy
import pandas as pd

#load spacy model
#Need to install spacy version >2.1
# and install vi_spacy
# using: https://github.com/trungtv/vi_spacy
nlp = spacy.load('vi_spacy_model')

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
                    nlp(q) for q, a in qa_list
        ]

        self.targets = targets
        self.qa_list = qa_list
    def retrieve_answer(self, question):
        '''
        Question:: String
        '''
        idoc = nlp(question)
        distances = [idoc.similarity(t) for t in self.targets]
        print(np.amax(distances))
        print(np.argmax(distances))
        if np.amax(distances)<0.8:
            id=0
        else:
            id = np.argmax(distances)

        #return most similar q & a
        return self.qa_list[id]

    def add_more_data(self,datafile):
        df = pd.read_csv(datafile, header=None, names=['intent', 'middle', 'q', 'a'])
        qa = df[['q', 'a']].copy()
        qa_list = qa[['q', 'a']].values
        targets = [
            nlp(q) for q, a in qa_list
        ]
        self.targets.extend(targets)
        self.qa_list.extend(qa_list)
if __name__ == '__main__':
    chitchat_file = 'data/chitchat.csv'
    chitchat = ChitChat(chitchat_file)
    
    # input_question = 'bạn thích gì nhất?'
    input_question = 'hiệu trưởng trường đại học công nghệ là ai'
    most_similar_question, answer = chitchat.retrieve_answer(input_question)
    
    print("inputquestion: ", input_question)
    print("most similar question found: ", most_similar_question)
    print("answer: ", answer)
    