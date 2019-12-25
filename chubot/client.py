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
# import speech_recognition as sr
from answer_retrieval import ChitChat
from custom_lib import changeUnicode
# from langdetect import detect
class ChatbotServer:
    def __init__(self):
        botname = "an"
        action_domain_file = "data/new_domain.json"
        self.action = ChuBotAction(botname)
        self.action.load_domain(action_domain_file)

        self.speak_code = 0
        self.go_around_code = 1
        self.lead_to_section_code = 2
        self.use_mp3_code = 3
        self.code = 0
        self.mp3 = -1
        self.section_id = -1
        ##load data to retrieve answer
        chitchat_file = 'data/chitchat.csv'
        ask_what_file = 'data/ask_what.csv'
        ask_who_file = 'data/ask_who.csv'
        ask_where_file = 'data/ask_where.csv'
        ask_number_file = 'data/ask_number.csv'
        ask_when_file = 'data/ask_when.csv'
        lead_to_section_file = 'data/lead_to_section.csv'
        self.answer_retriever = ChitChat(chitchat_file)
        self.answer_retriever.add_more_data(ask_what_file)
        self.answer_retriever.add_more_data(ask_who_file)
        self.answer_retriever.add_more_data(ask_where_file)
        self.answer_retriever.add_more_data(ask_number_file)
        self.answer_retriever.add_more_data(ask_when_file)
        self.answer_retriever.add_more_data(lead_to_section_file)
        self.nlp = spacy.load('vi_spacy_model')
    def create_model(self,name):
        self.action.chubot.load_data("data/full_train.json")
        self.meta = self.action.chubot.train()
    def test_entity_train(self,name):
        chubot = ChuBotBrain(name, language='vi')
        chubot.load_data("data/full_train.json")
        
        meta = chubot.train_nercrf()

    def encode_intent(self,intent):
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
        
    def test_intent_train(self,name):
        chubot = ChuBotBrain(name, language='vi')
        chubot.load_data("data/train.json")
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

    def test_response(self):
        botname = "an"
        action_domain_file = "data/action_domain.json"
        chubot = ChuBotAction(botname, language='vi')
        chubot.load_domain(action_domain_file)
        chubot.run_commandline_bot()

    def test_answer_retrieval(self,filename):
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
    ##
    # use to demo chatbot
    ##
    # def test_predict(self):

    #     ####Declare Speech Recognition
    #     r = sr.Recognizer()
    #     mic = sr.Microphone()
    #     with mic as source:
    #         while True:
    #             r.adjust_for_ambient_noise(source, 1)
    #             print("ready to record, please speak >> ")
    #             audio = r.listen(source,phrase_time_limit=5)
    #             try:
    #                 inmessage = r.recognize_google(audio,None,"vi-VN" "en-US")

    #                 #inmessage = input()
    #                 if inmessage == 'stop' or inmessage=='bye':
    #                     break
    #                 inmessage = inmessage.lower()
    #                 inmessage = changeUnicode.compound_unicode(inmessage)
    #                 print(inmessage)
                    # self.predict(inmessage)
    ##
    ## use to test chatbot
    ##
    def test_input_predict(self):
        nlp = spacy.load('vi_spacy_model')
        botname = "an"
        action_domain_file = "data/new_domain.json"
        section_id=-1
        response=""
        speak_code = 0
        lead_to_section_code = 1
        # chubot = ChuBotBrain(botname, language='vi')
        action = ChuBotAction(botname)
        action.load_domain(action_domain_file)
        intent_file = open('data/intent.csv', 'r', encoding="utf-8")
        csvreader = csv.reader(intent_file, delimiter=',')
        list_command_code = []
        for row in csvreader:
            list_command_code.append({"intent": row[1], "command_code": row[2]})
    
        while True:
            print("ready to record, please speak >> ")
            inmessage = input()
            if inmessage == 'stop' or inmessage=='bye':
                break
            inmessage = inmessage.lower()
            print(inmessage)
            self.predict(inmessage)
           
    #
    #use to demo on web and main predict, if fix format ouput or input, fix in here
    #
    def predict(self,inmessage):
        speak_code = 0
        go_around_code = 1
        lead_to_section_code = 2
        use_mp3_code = 3
        code = 0
        mp3 = -1
        section_id = -1
        inmessage = inmessage.lower()
        print(inmessage)
        responses = self.action.chubot.predict_intent(inmessage)
        entities = self.action.chubot.predict_entity(inmessage)
        (prob, intent) = responses[0]
        
        response = self.action.handle_message(inmessage)
        if(len(response)>0):
            response = response[0]
        else:
            response =""
        if intent=='chitchat':
            most_similar_question, answer = self.answer_retriever.retrieve_answer(inmessage,0)
            print(most_similar_question)
            response = answer
            ## open mp3 file to introduce the room
            if str(response) == "100.mp3":
                mp3 = 100
                code = use_mp3_code
        if intent=='ask_what':
            most_similar_question, answer = self.answer_retriever.retrieve_answer(inmessage,1)
            print(most_similar_question)
            response = answer
            if str(response) == "100.mp3":
                mp3 = 100
                code = use_mp3_code
        if intent=='ask_who':
            most_similar_question, answer = self.answer_retriever.retrieve_answer(inmessage,2)
            print(most_similar_question)
            response = answer
        if intent=='ask_where':
            most_similar_question, answer = self.answer_retriever.retrieve_answer(inmessage,3)
            print(most_similar_question)
            response = answer
        if intent=='ask_number':
            most_similar_question, answer = self.answer_retriever.retrieve_answer(inmessage,4)
            print(most_similar_question)
            response = answer
        if intent=='ask_when':
            most_similar_question, answer = self.answer_retriever.retrieve_answer(inmessage,5)
            print(most_similar_question)
            response = answer
        if intent =='command_lead_way' and len(entities)!=0:
            for entity in entities:
                if entity["entity"] =="section":
                    most_similar_question, answer = self.answer_retriever.retrieve_answer(inmessage,6)
                    print(most_similar_question)
                    section_id = answer
                    code = self.lead_to_section_code
                if entity["entity"] =="area":
                    code = self.go_around_code
            
        if intent=='ask_where' and len(entities)==0:
            mp3 = 15
            code = use_mp3_code
        if response == None:
            response = ""
        result_json = {"mp3":mp3,"section_id":section_id,
                    "code": code, "response": response}
        print(intent)
        print(entities)
        print(json.dumps(result_json, ensure_ascii=False))
        return result_json

if __name__ == "__main__":
    test = ChatbotServer()
    # test_entity_train('an')
    # test_intent_train('an')
    # test_answer_retrieval('data/chitchat_test.csv')
    # nlp = spacy.load('vi_spacy_model')

    ###############Retrain code################
    test.create_model('an')
    ###############Retrain code################

    ##########Server Code#################

    
    # test.test_input_predict()
    app = Flask(__name__)
    @app.route('/')
    def hello_world():
        if request.method == 'GET':
            mess = request.args.get('mess', '')
            if mess == None or mess=="":
                result_json = {"mp3":-1,"section_id":-1,
                    "code": 0, "response": ""}
                return str(result_json)    
            print(mess)
            line = test.predict(mess)
            return str(line).replace("'",'"')

    #     # return "null"

    app.run(host= '0.0.0.0')
    ########Server Code#################

    
    # test_input_predict()
