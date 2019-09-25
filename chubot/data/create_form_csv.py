import csv
def create_form_csv():
    Q_link = 'Q.csv'
    A_link = 'A.txt'
    rows =[]
    with open(Q_link,'r',encoding='utf-8') as Qin:
        reader = csv.reader(Qin, delimiter=',')
        for row in reader:
            rows.append(row)
    print('row',rows)
    # real_row = []
    # for row in rows:
    #     small_row = row.split('\r')
    #     real_row.extend(small_row)
    with open('Q_mod.csv','w',encoding='utf-8') as Qout:
        for row in rows:
            Qout.write(row[2])
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
    Q_link = 'Q.csv'
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

if __name__=='__main__':
    count_popular_word()
                    