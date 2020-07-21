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
botname = "an"
action_domain_file = "usingdata/new_domain.json"
action = ChuBotAction(botname)
action.load_domain(action_domain_file)

speak_code = 0
go_around_code = 1
lead_to_section_code = 2
use_mp3_code = 3
code = 0
mp3 = -1
section_id = -1
##load data to retrieve answer
chitchat_file = 'usingdata/chitchat.csv'
ask_what_file = 'usingdata/ask_what.csv'
ask_who_file = 'usingdata/ask_who.csv'
ask_where_file = 'usingdata/ask_where.csv'
ask_number_file = 'usingdata/ask_number.csv'
ask_when_file = 'usingdata/ask_when.csv'
lead_to_section_file = 'usingdata/lead_to_section.csv'
answer_retriever = ChitChat(chitchat_file)
answer_retriever.add_more_data(ask_what_file)
answer_retriever.add_more_data(ask_who_file)
answer_retriever.add_more_data(ask_where_file)
answer_retriever.add_more_data(ask_number_file)
answer_retriever.add_more_data(ask_when_file)
answer_retriever.add_more_data(lead_to_section_file)


nlp = spacy.load('vi_spacy_model')
def create_model(name):
    chubot = ChuBotBrain(name, language='vi')
    chubot.load_data("usingdata/full_train.json")
    # chubot.load_data("usingdata/vi_nlu_ask_way.json")
    meta = chubot.train()
    # print(meta)
    # print(chubot.entity_synonyms)

def test_entity_train(name):
    chubot = ChuBotBrain(name, language='vi')
    chubot.load_data("usingdata/full_train.json")
    # chubot.load_data("usingdata/vi_nlu_ask_way.json")
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
    if intent=='ask_number':
        return 7
    if intent=='ask_when':
        return 8
    if intent=='ask_location':
        return 9
    if intent=='end_conversation':
        return 10
    
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
    chitchat_file = 'usingdata/chitchat_train.csv'
    chitchat = ChitChat(chitchat_file)
    with open(filename,'r',encoding='utf-8')as f:
        rows = f.readlines()
    results = []
    origins = []
    for row in rows:
        parts = row.split(',')
        origin = parts[0]
        text = parts[1]
        most_similar_question, answer = chitchat.retrieve_answer(text,0)[0]
        results.append(most_similar_question)
        origins.append(origin)
    print('answer retriver precision ',precision_score(origins,results,average='weighted'))
##
# use to demo chatbot
##
def test_predict():
    ####Declare Speech Recognition
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        while True:
            r.adjust_for_ambient_noise(source, 1)
            print("ready to record, please speak >> ")
            audio = r.listen(source,phrase_time_limit=4)
            try:
                inmessage = r.recognize_google(audio,None,"vi-VN" "en-US")

                #inmessage = input()
                if inmessage == 'stop' or inmessage=='bye':
                    break
                inmessage = inmessage.lower()
                inmessage = changeUnicode.compound_unicode(inmessage)
                print(inmessage)
                predict(inmessage)

            except sr.UnknownValueError:

                print("Oops! Didn't catch that ")
            except sr.RequestError as e:
                print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))

##
## use to test chatbot
##
def test_input_predict():
    while True:
        print("ready to record, please speak >> ")
        inmessage = input()
        if inmessage == 'stop' or inmessage=='bye':
            break
        inmessage = inmessage.lower()
        print(inmessage)
        predict(inmessage)
#
#use to demo on web
#
def predict(inmessage):
    speak_code = 0
    go_around_code = 1
    lead_to_section_code = 2
    use_mp3_code = 3
    code = 0
    mp3 = -1
    section_id = -1

    inmessage = inmessage.lower()
    print(inmessage)
    responses = action.chubot.predict_intent(inmessage)
    entities = action.chubot.predict_entity(inmessage)
    # for entity in entities:
    #     if entity['entity'] == 'present' and entity['confidence'] > 0.85:
    #         mp3 = 100
    #         code = use_mp3_code

    (prob, intent) = responses[0]
    response = action.handle_message(inmessage)[0]
    if intent=='chitchat':
        most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,0)[0]
        print(most_similar_question)
        response = answer
        ## open mp3 file to introduce the room
        if str(response) == "100.mp3":
            mp3 = 100
            code = use_mp3_code
    if intent=='ask_what':
        most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,1)[0]
        print(most_similar_question)
        response = answer
        if str(response) == "100.mp3":
            mp3 = 100
            code = use_mp3_code
        for entity in entities:
            if entity['entity'] == 'present' and entity['confidence'] > 0.85:
                mp3 = 100
                code = use_mp3_code
                response = "100.mp3"
    if intent=='ask_who':
        most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,2)[0]
        print(most_similar_question)
        response = answer
    if intent=='ask_where':
        most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,3)[0]
        print(most_similar_question)
        response = answer
    if intent=='ask_number':
        most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,4)[0]
        print(most_similar_question)
        response = answer
    if intent=='ask_when':
        most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,5)[0]
        print(most_similar_question)
        response = answer
    if intent =='command_lead_way' and len(entities)!=0:
        for entity in entities:
            if entity["entity"] =="section":
                most_similar_question, answer = answer_retriever.retrieve_answer(inmessage,6)[0]
                print(most_similar_question)
                section_id = answer
                code = lead_to_section_code
            if entity["entity"] =="area":
                code = go_around_code
        
    if intent=='ask_where' and len(entities)==0:
        mp3 = 15
        code = use_mp3_code
    result_json = {"mp3":mp3,"section_id":section_id,
                "code": code, "response": response}
    
    print(intent)
    print(entities)
    
    print(json.dumps(result_json, ensure_ascii=False))
    return result_json

if __name__ == "__main__":
    # test_entity_train('an')
    # test_intent_train('an')
    # test_answer_retrieval('usingdata/chitchat_test.csv')
    # nlp = spacy.load('vi_spacy_model')

    ###############Retrain code################
    # create_model('an')
    ###############Retrain code################

    ##########Server Code#################

    # hãy giới thiệu về đại học quốc gia
    # app = Flask(__name__)
    # @app.route('/')
    # def hello_world():
    #     if request.method == 'GET':
    #         mess = request.args.get('mess', '')
    #         print(mess)
    #         line = predict(mess)
    #         # print(str(line))
    #         return str(line).replace("'",'"')

    # #     # return "null"

    # app.run(host= '0.0.0.0')
    ##########Server Code#################

    # test_predict()
    test_input_predict()
