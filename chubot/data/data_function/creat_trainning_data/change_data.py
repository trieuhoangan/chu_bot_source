# import pandas as pd
import json
import csv
import spacy
import random
import pandas as pd

json_object = {"nlu_data": {"common_examples": [],
                                "regex_features": [], "lookup_tables": [], "entity_synonyms": []}}
intent_train_set = []
usable_set = []
###
#
#
#
#
###
class DataObject():
    def __init__(self):
        self.count = 0
        self.json_object = {"nlu_data": {"common_examples": [],
                                "regex_features": [], "lookup_tables": [], "entity_synonyms": []}}
    #####
    #     use to load all entity that have been confirmed
    ####
    def load_entity_data(self,entity_link):
        entity_file = open(entity_link, newline='', encoding="utf8")
        csv_reader = csv.reader(entity_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                symnonyms = row[2].split(',')
                self.json_object.get("nlu_data").get("entity_synonyms").append(
                    {"entity": row[0], "original_value": row[1], "synonyms": symnonyms})

            line_count = line_count+1
        entity_file.close()
    ###s
    #    load data from .txt file with data split by ','
    ###
    def load_quest_data(self,quest_link,text_col,intent_col):
        intent_file = open(quest_link,'r', encoding="utf8")
        csv_reader = intent_file.readlines()
        # nlp = spacy.load('vi_core_news_md')
        all_entity = self.json_object.get("nlu_data").get("entity_synonyms")
        line_count = 0
        for line in csv_reader:
            if line_count == 0:
                line_count = line_count+1
                continue
            row = line.split(',')
            text = row[text_col].lower().replace('\n','')
            intent = row[intent_col].lower()
            if text != "":
                entities = []

                for entity in all_entity:
                    for symnonym in entity.get("synonyms"):
                        if symnonym in text:
                            start = text.index(symnonym)
                            if start > 1 and text[start-1] != ' ':
                                continue
                            
                            end = start + len(symnonym) - 1
                            if end+1 != len(text):
                                if text[end+1] != ' ':
                                    continue
                            entities.append({"start": start, "end": end, "entity": entity.get(
                                "entity"), "value": entity.get("original_value")})
                if len(entities) == 0:
                    self.json_object.get("nlu_data").get("common_examples").append(
                        {"text": text, "intent": intent})
                else:
                    self.json_object.get("nlu_data").get("common_examples").append(
                        {"text": text, "intent": intent, "entities": entities})
        intent_file.close()
    ###
    # load data from csv and remove duplicate data
    # load sentences and the label of sentences only
    # load entities and the position on the sentences
    ###
    def load_distinc_data(self,filename,text_col,intent_col,limit_sentence):
        "first param is text, second param is intent"
        intent_file = open(filename, newline='', encoding="utf8")

        # nlp = spacy.load('vi_core_news_md')
        if limit_sentence < 1:
            limit_sentence = 100000
        all_entity = self.json_object.get("nlu_data").get("entity_synonyms")
        csv_reader = csv.reader(intent_file, delimiter=',')
        lines = []
        examples = []
        line_count = 0
        count =0
        # read data from *.csv file and ignore duclicated data
        for row in csv_reader:
            if line_count == 0:
                line_count = line_count+1
                continue
            if count > limit_sentence:
                break
            else:
                count = count +1 
            text = row[text_col].lower()
            intent = row[intent_col].lower()
            if text !='':
                if text not in lines:
                    self.count = self.count+1
                    lines.append(text)
                    examples.append({'text':text,'intent':intent})
        
        # put data into json object as the format
        for example in examples:
            
            entities = []
            for entity in all_entity:
                for symnonym in entity.get("synonyms"):
                    if symnonym in example.get('text'):
                        start = example.get('text').index(symnonym)
                        if start > 1 and example.get('text')[start-1] != ' ':
                            continue

                        end = start + len(symnonym) - 1
                        if end+1 != len(example.get('text')):
                            if example.get('text')[end+1] != ' ':
                                continue
                        entities.append({"start": start, "end": end, "entity": entity.get(
                            "entity"), "value": entity.get("original_value")})
            if len(entities) == 0:
                self.json_object.get("nlu_data").get("common_examples").append(
                    {"text": example.get('text'), "intent": example.get('intent')})
            else:
                self.json_object.get("nlu_data").get("common_examples").append(
                    {"text": example.get('text'), "intent": example.get('intent'), "entities": entities})
        
        intent_file.close()
    ###
    # split test-train dataset from a list of files 
    ###
    def split_train_test_dataset(self,list_filename,intent_col,text_col,test_ratio,test_output,train_output):
        dataset = { 'ask_what':[],
                    'ask_who':[],
                    'ask_where':[],
                    'greeting':[],
                    'ask_location':[],
                    'chitchat':[],
                    'command_lead_way':[],
                    'end_conversation':[],
                    'ask_number':[],
                    'ask_when':[],
                    'unknown':[]
                    }
        intents = []
        lines =[]
        # read data from csv file
        for filename in list_filename:
            count = 0
            print("filename",filename)
            with open(filename,'r',encoding='utf-8') as fin:
                rows = fin.readlines()
            for row in rows:
                parts = row.split(',')
                intent = parts[intent_col]
                # print(parts)
                print("number",count)
                text = parts[text_col].lower().replace('\n','')
                count = count+1
                if text not in lines:
                    lines.append(text)
                    dataset.get(intent).append({'intent':intent,'text':text,'a':parts[1]})
                if intent not in intents:
                    intents.append(intent)
            with open("line.txt",'a',encoding='utf-8') as fline:
                fline.writelines(lines)
        print(intents)
        #######
        train_set = []
        test_set = []
        # split dataset
        for intent in intents:
            datalength = len(dataset.get(intent))
            random.shuffle(dataset.get(intent))
            train_set.extend(dataset.get(intent)[int(datalength*test_ratio):])
            test_set.extend(dataset.get(intent)[:int(datalength*test_ratio)])
        full_train_set = []
        full_train_set.extend(train_set)
        full_train_set.extend(test_set)

        ########### write data into .json file
        # full_train_link = 'full_train.txt'
        
        test_link = test_output
        train_link = train_output
        with open(test_link,'w',encoding='utf-8') as test_out:
            for test in test_set:
                test_out.write('{},{},{}\n'.format(test.get('intent'),test.get('text'),test['a']))
        with open(train_link,'w',encoding='utf-8') as train_out:
            for train in train_set:
                train_out.write('{},{},{}\n'.format(train.get('intent'),train.get('text'),train['a']))
        # with open(full_train_link,'w',encoding='utf-8') as test_out:
        #     for test in full_train_set:
        #         test_out.write('{},{}\n'.format(test.get('intent'),test.get('text')))
    
    # use to write data as the noised data to recognise noise and good data
    def printCount(self):
        print(self.count)
    def split_train_test_chitchat(self,list_filename,origin_col,noice_col,test_output,train_output):
        dataset = []
        test = []
        lines =[]
        test_line=[]
        #read data
        for filename in list_filename:
            count = 0
            print("filename",filename)
            with open(filename,'r',encoding='utf-8') as fin:
                rows = fin.readlines()
            for row in rows:
                parts = row.split(',')
                origin = parts[origin_col]
                # print(parts)
                print("number",count)
                noice = parts[noice_col].lower().replace('\n','')
                a = parts[3].lower().replace('\n','')
                count = count+1
                if origin not in lines:
                    lines.append(origin)
                    dataset.append({'q':origin,'a':a})
                if noice not in lines and noice not in test_line:
                    test_line.append(noice)
                    test.append({'o':origin,'q':noice,'a':a})
            with open("line.txt",'a',encoding='utf-8') as fline:
                fline.writelines(lines)


        train_set = dataset
        test_set = test

        test_link = test_output
        train_link = train_output
        with open(test_link,'w',encoding='utf-8') as test_out:
            for test in test_set:
                test_out.write('{},{},{}\n'.format(test['o'],test['q'],test['a']))
        i = 0
        with open(train_link,'w',encoding='utf-8') as train_out:
            for train in train_set:
                i = i +1
                train_out.write('{},{},a,{}\n'.format(i,train['q'],train['a']))
                
        
    def make_chit_chat_train_test(self,filename,intent_col):
        df = pd.read_csv(filename, header=None, names=['intent', 'q', 'noisyq', 'a'])
    



if __name__=="__main__":
    # list_filename=['chitchat.csv','greeting_end.csv','command_lead_way.csv','ask_robot.csv']
    train_link = 'train.json'
    test_link = 'test.json'
    # list_filename=['../chitchat.csv','../command_lead_way.csv',"../ask_robot.csv",'../greeting_end.csv']
    a = DataObject()
    # a.split_train_test_dataset(list_filename,0,2,1,train_link,test_link)
    a.load_entity_data('./entity_list.csv')
    a.load_distinc_data('./chitchat.csv',1,0,-1)
    a.load_distinc_data('./command_lead_way.csv',2,0,200)
    # a.load_distinc_data('./ask_robot.csv',2,0,0)
    a.load_distinc_data('./ask_who.csv',1,0,-1)
    a.load_distinc_data('./ask_where.csv',1,0,-1)
    a.load_distinc_data('./ask_what.csv',1,0,200)
    a.load_distinc_data('./ask_when.csv',1,0,-1)
    a.load_distinc_data('./ask_number.csv',1,0,200)
    a.load_distinc_data('./ask_when.csv',1,0,-1)
    a.load_distinc_data('./greeting_end.csv',2,0,-1)
    a.load_distinc_data('./presentation.csv',1,0,-1)
    a.load_distinc_data('./command.csv',1,0,-1)
    a.load_distinc_data('./presentation2.csv',1,0,-1)
    a.printCount()
    # a.load_quest_data('../oldata/quest_data.csv',2,0)
    # a.load_distinc_data('quest_data_noise.csv',2,0)
    # load_quest_data('Q.csv',2,0)
    
    # b.split_train_test_dataset(list_filename,0,2,0.3)
    # b.load_entity_data('entity_list.csv')
    # a.load_quest_data('train.txt',1,0)

    # link_ouput = 'test.json'
    # with open(link_ouput, 'w', encoding='utf8') as output:
    #     output.write(json.dumps(a.json_object, ensure_ascii=False))
    #     print('done')

    # link_ouput = 'chitchat.json'
    # with open(link_ouput, 'w', encoding='utf8') as output:
    #     output.write(json.dumps(a.json_object, ensure_ascii=False))
    #     print('done')
    # b = DataObject()
    # b.load_quest_data('full_train.txt',1,0)
    full_train_output = './full_train.json'
    # full_train_output = './part.json'
    with open(full_train_output, 'w', encoding='utf8') as output:
        output.write(json.dumps(a.json_object, ensure_ascii=False))
        print('done')
