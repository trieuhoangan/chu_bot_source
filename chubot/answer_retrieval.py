import numpy as np
import spacy
import pandas as pd
import heapq
import datetime
from gensim.models import KeyedVectors
import os
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
        self.dirname = os.path.dirname(__file__)
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
        word2vecmodel_link = self.dirname + "/models/wiki.vi.model.bin"
        self.word2vec = KeyedVectors.load_word2vec_format(fname=word2vecmodel_link,binary=True,unicode_errors='strict')
        self.targets = []
        self.targets.append(targets)
        self.qa_list = []
        self.qa_list.append(qa_list)
        

    def retrieve_answer(self, question,ids):
        '''
        Question:: String
        '''
        # print("this is the bank {}".format(self.qa_list))
        question = question + " "
        digits = [" 11 "," 10 "," 9 "," 8 "," 7 "," 6 "," 5 "," 4 "," 3 ",' 2 ',' 1 ']
        text_digits = [' mười một',' mười',' chín',' tám',' bảy',' sáu',' năm',' bốn',' ba',' hai',' một']
        for digit_ in digits:
            if digit_ in question:
                question = question.replace(digit_, text_digits[digits.index(digit_)])
                break
        start = datetime.datetime.now()
        distances = [self.word2vec.wmdistance(question.split(" "),t.split(" ")) for t in self.targets[ids]]
        id_result=np.argmin(distances)
        print(distances[id_result])
        print("answer",self.qa_list[ids][id_result])
        if distances[id_result] >29:
            return [self.qa_list[ids][2],self.qa_list[ids][2],self.qa_list[ids][2]]
        if distances[id_result] >28:
            return [self.qa_list[ids][1],self.qa_list[ids][1],self.qa_list[ids][1]]
            # return self.qa_list[ids][0]
        # print(id)
        # top3 = heapq.nsmallest(3,range(len(distances)),distances.__getitem__)
        # print(top3)
        # list_answer = []
        # for id_ans in top3:
        #     list_answer.append(self.qa_list[ids][id_ans])
        # return list_answer
        end = datetime.datetime.now()
        print("retrieval time : ",end-start)
        return [self.qa_list[ids][id_result],self.qa_list[ids][id_result],self.qa_list[ids][id_result]]
    def retrieve_section(self, question,ids):
        digits = [" 11"," 10"," 9"," 8"," 7"," 6"," 5"," 4"," 3",' 2',' 1']
        text_digits = [' mười một',' mười',' chín',' tám',' bảy',' sáu',' năm',' bốn',' ba',' hai',' một']
        for digit_ in digits:
            if digit_ in question:
                if digit_ == " 10":
                    return [(' khu vực số mười','10')]
                if digit_ ==" 1":
                    return [(' khu vực số một','1')]    
                question = question.replace(digit_, text_digits[digits.index(digit_)])
                break
        start = datetime.datetime.now()
        distances = [self.word2vec.wmdistance(question.split(" "),t.split(" ")) for t in self.targets[ids]]
        id_result=np.argmin(distances)
        print(distances[id_result])
        print("answer",self.qa_list[ids][id_result])
        if distances[id_result] >35:
            return [self.qa_list[ids][2],self.qa_list[ids][2],self.qa_list[ids][2]]
        if distances[id_result] >32:
            return [self.qa_list[ids][1],self.qa_list[ids][1],self.qa_list[ids][1]]
            # return self.qa_list[ids][0]
        # print(id)
        # top3 = heapq.nsmallest(3,range(len(distances)),distances.__getitem__)
        # print(top3)
        # list_answer = []
        # for id_ans in top3:
        #     list_answer.append(self.qa_list[ids][id_ans])
        # return list_answer
        end = datetime.datetime.now()
        print("retrieval time : ",end-start)
        return [self.qa_list[ids][id_result],self.qa_list[ids][id_result],self.qa_list[ids][id_result]]
    def add_more_data(self,datafile):
        df = pd.read_csv(datafile, header=None, names=['intent', 'q', 'n', 'a'])
        qa = df[['q', 'a']].copy()
        qa_list = qa[['q', 'a']].values
        targets = [
            q for q, a in qa_list
        ]
        self.targets.append(targets)
        self.qa_list.append(qa_list)
        # print("this is the bank {}".format(self.qa_list))
        # print("this is the target {}".format(self.targets))
    def print_all_data(self):
        outfile_link="data_of_ar.csv"
        output_stream = open(outfile_link,'w',encoding='utf-8')
        i = 16
        for dataset in self.qa_list:
            for sentence in dataset:
                q,a = sentence
                if q !='q':
                    output_stream.write("{},{},{}\n".format(i,q,a))
                    i = i + 1
        output_stream.close()
if __name__ == '__main__':
    chitchat_file = '/data/chitchat.csv'
    chitchat = ChitChat(chitchat_file)
    
    # input_question = 'bạn thích gì nhất?'
    input_question = 'hiệu trưởng trường đại học công nghệ là ai'
    most_similar_question, answer = chitchat.retrieve_answer(input_question,0)[0]
    
    print("inputquestion: ", input_question)
    print("most similar question found: ", most_similar_question)
    print("answer: ", answer)
    