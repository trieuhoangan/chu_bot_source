import pandas as pd
import csv
def create_form_csv():
    Q_link = '../Q.csv'
    A_link = '../A.csv'
    Qrows =[]
    Arows =[]
    rows =[]
    with open(Q_link,'r',encoding='utf-8') as Qin:
        Qreader = csv.reader(Qin, delimiter=',')
        for row in Qreader:
            Qrows.append(row)
    with open(A_link,'r',encoding='utf-8') as Ain:
        Areader = csv.reader(Ain, delimiter=',')
        for row in Areader:
            Arows.append(row)
    print("Qlen",len(Qrows))
    print("Alen",len(Arows))

    for i in range(len(Qrows)):
        Qrows[i].extend("b")
        Qrows[i].extend(Arows[i])
    print(Qrows)
    with open("../QA.csv",'w',encoding='utf-8') as QAout:
        for row in Qrows:
            QAout.write(str(row)+"\n")

def sort_dict(dictionary):
    print('sorting')
    sorted_dict = []
    while(len(dictionary)!=0):
        max = 0
        for word in dictionary:
            if word['number'] >max:
                max = word['number']
                tmp = word
        sorted_dict.append(tmp)
        dictionary.remove(tmp)
    return sorted_dict
def count_popular_word():
    import spacy 
    nlp = spacy.load('vi_spacy_model')
    Q_link = '../Q.csv'
    total_quest = []
    dictionary = []
    with open(Q_link,'r',encoding='utf-8') as Qin:
        rows = Qin.readlines()
        for row in rows:
            parts = row.split(',')
            total_quest.append(parts[2].lower())
    for quest in total_quest:
        for token in nlp(quest):
            if(len(dictionary)==0):
                dictionary.append({'word':token.text,'number':1})
            else :
                existed = 0
                for word in dictionary:
                    if word['word'] == token.text:
                        word['number'] = word['number'] +1
                        existed = existed + 1
                        break
                if existed==0:
                    dictionary.append({'word':token.text,'number':1})
    dictionary = sort_dict(dictionary)
    with open('dict.txt','w',encoding='utf-8') as f:
        for word in dictionary:
            f.write('{},{}\n'.format(word['word'],word['number']))
    print('dic:',dictionary)
def join_Q_A(datafile):
    df = pd.read_csv(datafile, header=None, names=['intent', 'mid', 'q'])
    af = pd.read_csv("../A.txt",header=None,names=['a'])
    a = af[['a']].copy()
    qa = df[['intent','q']].copy()
    print(qa[['intent','q']].values+a[['a']].values)
   
if __name__=='__main__':
    create_form_csv()
                    