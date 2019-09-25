# import pandas as pd
import json
import csv
import spacy
import random


# for entity in  json_object.get("nlu_data").get("entity_synonyms"):
#     new_synonyms = []
#     for synonym in entity.get("synonyms"):
#         synonym_token = [token.text for token in nlp(synonym)]
#         new_synonyms.append(" ".join(synonym_token))
#     entity.__setitem__("synonyms",new_synonyms)
json_object = {"nlu_data": {"common_examples": [],
                                "regex_features": [], "lookup_tables": [], "entity_synonyms": []}}
intent_train_set = []
usable_set = []
class DataObject():
    def __init__(self):
        self.json_object = {"nlu_data": {"common_examples": [],
                                "regex_features": [], "lookup_tables": [], "entity_synonyms": []}}
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
    # load data from .txt file with data split by ','
    #
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
    ###
    def load_distinc_data(self,filename,intent_col,entity_col):
        intent_file = open(filename, newline='', encoding="utf8")

        # nlp = spacy.load('vi_core_news_md')
        all_entity = self.json_object.get("nlu_data").get("entity_synonyms")
        csv_reader = csv.reader(intent_file, delimiter=',')
        lines = []
        examples = []
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count = line_count+1
                continue
            text = row[intent_col].lower()
            intent = row[entity_col].lower()
            if text !='':
                if text not in lines:
                    lines.append(text)
                    examples.append({'text':text,'intent':intent})
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
    #
    ###
    def split_train_test_dataset(self,list_filename,intent_col,text_col,test_ratio):
        dataset = { 'ask_what':[],
                    'ask_who':[],
                    'ask_number':[],
                    'ask_where':[],
                    'greeting':[],
                    'introduce':[],
                    'chat':[],
                    'command_lead_way':[],
                    'introduce_vnu':[],
                    'end_conversation':[],
                    'ask_when':[]
                    }
        intents = []
        lines =[]
        for filename in list_filename:
            with open(filename,'r',encoding='utf-8') as fin:
                rows = fin.readlines()
            for row in rows:
                parts = row.split(',')
                intent = parts[intent_col]
                # print(parts)
            
                text = parts[text_col].lower().replace('\n','')
                if text not in lines:
                    lines.append(text)
                    dataset.get(intent).append({'intent':intent,'text':text})
                if intent not in intents:
                    intents.append(intent)
            with open("line.txt",'a',encoding='utf-8') as fline:
                fline.writelines(lines)
        print(intents)
        train_set = []
        test_set = []
        for intent in intents:
            datalength = len(dataset.get(intent))
            random.shuffle(dataset.get(intent))
            train_set.extend(dataset.get(intent)[int(datalength*test_ratio):])
            test_set.extend(dataset.get(intent)[:int(datalength*test_ratio)])
        
        test_link = 'test.txt'
        train_link = 'train.txt'
        with open(test_link,'w',encoding='utf-8') as test_out:
            for test in test_set:
                test_out.write('{},{}\n'.format(test.get('intent'),test.get('text')))
        with open(train_link,'w',encoding='utf-8') as train_out:
            for train in train_set:
                train_out.write('{},{}\n'.format(train.get('intent'),train.get('text')))
    
if __name__=="__main__":
    list_filename=['quest_data_noise.csv','Q.csv']
    a = DataObject()
    a.split_train_test_dataset(list_filename,0,2,0)
    a.load_entity_data('entity_list.csv')
    # a.load_quest_data('test.txt',1,0)
    # a.load_distinc_data('quest_data_noise.csv',2,0)
    # load_quest_data('Q.csv',2,0)
    # b = DataObject()
    # b.split_train_test_dataset(list_filename,0,2,0.3)
    # b.load_entity_data('entity_list.csv')
    a.load_quest_data('train.txt',1,0)
    # link_ouput = 'test.json'
    # with open(link_ouput, 'w', encoding='utf8') as output:
    #     output.write(json.dumps(a.json_object, ensure_ascii=False))
    #     print('done')
    link_ouput = 'train.json'
    with open(link_ouput, 'w', encoding='utf8') as output:
        output.write(json.dumps(a.json_object, ensure_ascii=False))
        print('done')
