# coding=<utf-8>
from flask import Flask
from chubot_brain import ChuBotBrain
from chubot_action import ChuBotAction
from api import ChatBotAPI
import sys
import csv
import json
import codecs
from flask import request
import spacy
from sklearn.metrics import accuracy_score
# from langdetect import detect


def create_model(name):
    chubot = ChuBotBrain(name, language='vi')
    chubot.load_data("data/train.json")
    # chubot.load_data("data/vi_nlu_ask_way.json")
    meta = chubot.train()
    # print(meta)
    # print(chubot.entity_synonyms)

def test_entity_train(name):
    chubot = ChuBotBrain(name, language='vi')
    chubot.load_data("data/full_train.json")
    # chubot.load_data("data/vi_nlu_ask_way.json")
    meta = chubot.train_nercrf()

def encode_intent(intent):
    if intent=='greeting':
        return 1
    if intent=='chitchat':
        return 2
    if intent=='command_lead_way':
        return 3
    if intent=='ask_who':
        return 4    
    if intent=='ask_what':
        return 5
    if intent=='ask_where':
        return 6
    if intent=='ask_location':
        return 7
    if intent=='end_conversation':
        return 8
    
def test_intent_train(name):
    chubot = ChuBotBrain(name, language='vi')
    chubot.load_data("data/train.json")
    # chubot.load_data("data/vi_nlu_ask_way.json")
    meta = chubot.train_intent_classification()
    test_link = "data/test.txt"
    with open(test_link,'r',encoding='utf=8') as f:
        rows = f.readlines()
    intent_list_test = []
    intent_list_result = []
    data_list_test = []
    for row in rows:
        parts = row.split(',')
        intent_list_test.append(encode_intent(parts[0]))
        data_list_test.append(parts[1])
    for data in data_list_test:
        responses = chubot.predict_intent(data)
        (prob, intent) = responses[0]
        intent_list_result.append(encode_intent(intent))
    print("accuracy",accuracy_score(intent_list_test,intent_list_result))

def test_response():
    botname = "an"
    action_domain_file = "data/action_domain.json"
    chubot = ChuBotAction(botname, language='vi')
    chubot.load_domain(action_domain_file)
    chubot.run_commandline_bot()

def test_predict():
    nlp = spacy.load('vi_spacy_model')
    botname = "an"
    action_domain_file = "data/new_domain.json"

    # chubot = ChuBotBrain(botname, language='vi')
    action = ChuBotAction(botname)
    action.load_domain(action_domain_file)
    intent_file = open('data/intent.csv', 'r', encoding="utf-8")
    csvreader = csv.reader(intent_file, delimiter=',')
    list_command_code = []
    for row in csvreader:
        list_command_code.append({"intent": row[1], "command_code": row[2]})

    while True:
        print("you >")
        inmessage = input()
        if inmessage == 'stop':
            break
        inmessage = inmessage.lower()

        responses = action.chubot.predict_intent(inmessage)
        entities = action.chubot.predict_entity(inmessage)
        
        # print(responses)
        print("\n")
        print(entities)
        (prob, intent) = responses[0]
        # print(prob)
        print(intent)
        command_code = 0
        mp3 = -1
        for command in list_command_code:
            if(command.get("intent") == intent):
                print(command.get("command_code"))
                command_code = command.get("command_code")
        response = action.handle_message(inmessage)
        if intent=='ask_where' and len(entities)==0:
            mp3 = 15
        if intent=='introduce_vnu':
            mp3 = int(response[0])
        result_json = {"intent": intent, "entities": entities,"mp3":mp3,
                       "command_code": command_code, "response": response}
        print(json.dumps(result_json, ensure_ascii=False))
        for token in nlp(inmessage):
            if token.tag_ =='P' or token.tag_=='Np':
                print(token.text)
        print("bot > "+str(response))
        # print(result_json)
def get_data():
    quest_link = 'data/Q.csv'
    answer_link = 'data/A.txt'
    with open(quest_link,'r',encoding='utf-8') as qin:
        questions = qin.readlines()
    with open(answer_link,'r',encoding='utf-8') as ain:
        answers = ain.readlines()
    data = {'question':questions,'answer':answers}
    return data
if __name__ == "__main__":
    test_entity_train('an')
    test_intent_train('an')
    # nlp = spacy.load('vi_spacy_model')
    # create_model('an')
    # data = get_data()
    # botname= 'an'
    # action = ChuBotAction(botname)
    
    # app = Flask(__name__)
    # @app.route('/')
    # def hello_world():
    #     if request.method == 'GET':
    #         mess = request.args.get('mess', '')

    #         bot = ChatBotAPI('vi', 'an')
    #         bot.load_model()
    #         line = bot.predict_message(mess)
    #         return line

    #     # return "null"

    # app.run()

   
    # test_predict(extracted_question,data.get('answer'))
    # test_predict()
    # # test_response()
    # bot = ChatBotAPI('vi', 'an')
    # bot.load_model()
    
    # line = bot.predict_message("who are you")
    # print(line)
    # print(detect("War doesn't show who's right, just who's left."))
    # detect("War doesn't show who's right, just who's left.")
   
