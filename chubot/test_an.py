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
from sklearn.metrics import precision_score
import speech_recognition as sr
from answer_retrieval import ChitChat
from custom_lib import changeUnicode
# from langdetect import detect


def create_model(name):
    chubot = ChuBotBrain(name, language='vi')
    chubot.load_data("data/newQAdata/train.json")
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

def test_answer_retrieval(filename):
    chitchat_file = 'data/chitchat_train.csv'
    chitchat = ChitChat(chitchat_file)
    with open(filename,'r',encoding='utf-8')as f:
        rows = f.readlines()
    results = []
    origins = []
    for row in rows:
        parts = row.split(',')
        origin = parts[0]
        text = parts[1]
        most_similar_question, answer = chitchat.retrieve_answer(text)
        results.append(most_similar_question)
        origins.append(origin)
    print('answer retriver precision ',precision_score(origins,results,average='weighted'))

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
    chitchat_file = 'data/chitchat.csv'
    ask_what_file = 'data/ask_what.csv'
    ask_who_file = 'data/ask_who.csv'
    ask_where_file = 'data/ask_where.csv'
    ask_number_file = 'data/ask_number.csv'
    ask_when_file = 'data/ask_when.csv'
    answer_retriever = ChitChat(chitchat_file)
    answer_retriever.add_more_data(ask_what_file)
    answer_retriever.add_more_data(ask_who_file)
    answer_retriever.add_more_data(ask_where_file)
    answer_retriever.add_more_data(ask_number_file)
    answer_retriever.add_more_data(ask_when_file)
    ##chitchat.add_more_data("data/QA.csv")


    ####Declare Speech Recognition
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        while True:
            r.adjust_for_ambient_noise(source, 1)
            print("ready to record, please speak >> ")
            audio = r.listen(source,phrase_time_limit=2)
            try:
                inmessage = r.recognize_google(audio,None,"vi-VN" "en-US")

                #inmessage = input()
                if inmessage == 'stop' or inmessage=='bye':
                    break
                inmessage = inmessage.lower()
                inmessage = changeUnicode.compound_unicode(inmessage)
                print(inmessage)
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
                if intent=='chitchat':
                    most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,0)
                    print(most_similar_question)
                    response = answer
                if intent=='ask_what':
                    most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,1)
                    print(most_similar_question)
                    response = answer
                if intent=='ask_who':
                    most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,2)
                    print(most_similar_question)
                    response = answer
                if intent=='ask_where':
                    most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,3)
                    print(most_similar_question)
                    response = answer
                if intent=='ask_number':
                    most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,4)
                    print(most_similar_question)
                    response = answer
                if intent=='ask_when':
                    most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,5)
                    print(most_similar_question)
                    response = answer
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
                    
            except sr.UnknownValueError:
                text = r.recognize_google(audio,None,"vi-VN" "en-US",show_all=True)
                # print("Oops! Didn't catch that ")
                print(text)
            except sr.RequestError as e:
                print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))

def test_input_predict():
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
    
    chitchat_file = 'data/chitchat.csv'
    ask_what_file = 'data/ask_what.csv'
    ask_who_file = 'data/ask_who.csv'
    ask_where_file = 'data/ask_where.csv'
    ask_number_file = 'data/ask_number.csv'
    ask_when_file = 'data/ask_when.csv'
    answer_retriever = ChitChat(chitchat_file)
    answer_retriever.add_more_data(ask_what_file)
    answer_retriever.add_more_data(ask_who_file)
    answer_retriever.add_more_data(ask_where_file)
    answer_retriever.add_more_data(ask_number_file)
    answer_retriever.add_more_data(ask_when_file)
    ##chitchat.add_more_data("data/QA.csv")


    
    while True:
        print("ready to record, please speak >> ")
        inmessage = input()
        if inmessage == 'stop' or inmessage=='bye':
            break
        inmessage = inmessage.lower()
        print(inmessage)
        responses = action.chubot.predict_intent(inmessage)
        entities = action.chubot.predict_entity(inmessage)
        
        print("\n")
        print(entities)
        (prob, intent) = responses[0]
        print(intent)
        command_code = 0
        mp3 = -1
        for command in list_command_code:
            if(command.get("intent") == intent):
                print(command.get("command_code"))
                command_code = command.get("command_code")
        response = action.handle_message(inmessage)
        if intent=='chitchat':
            most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,0)
            print(most_similar_question)
            response = answer
        if intent=='ask_what':
            most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,1)
            print(most_similar_question)
            response = answer
        if intent=='ask_who':
            most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,2)
            print(most_similar_question)
            response = answer
        if intent=='ask_where':
            most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,3)
            print(most_similar_question)
            response = answer
        if intent=='ask_number':
            most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,4)
            print(most_similar_question)
            response = answer
        if intent=='ask_when':
            most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,5)
            print(most_similar_question)
            response = answer
        
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
if __name__ == "__main__":
    # test_entity_train('an')
    # test_intent_train('an')
    # test_answer_retrieval('data/chitchat_test.csv')
    # nlp = spacy.load('vi_spacy_model')

    ###############Retrain code################
    # create_model('an')
    ###############Retrain code################

    ##########Server Code#################
    # botname= 'an'
    # action = ChuBotAction(botname)
    
    # app = Flask(__name__)
    # @app.route('/')
    # def hello_world():
    #     if request.method == 'GET':
    #         mess = request.args.get('mess', '')
    #         print(mess)
    #         bot = ChatBotAPI('vi', 'an')
    #         bot.load_model()
    #         line = bot.predict_message(mess)
    #         return line

    # #     # return "null"

    # app.run(host= '0.0.0.0')
    ##########Server Code#################

    # test_predict()
    test_input_predict()
