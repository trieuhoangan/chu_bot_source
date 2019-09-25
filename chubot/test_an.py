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
# from langdetect import detect


def create_model(name):
    chubot = ChuBotBrain(name, language='vi')
    chubot.load_data("data/train.json")
    # chubot.load_data("data/vi_nlu_ask_way.json")
    meta = chubot.train()
    # print(meta)
    # print(chubot.entity_synonyms)


def test_response():
    botname = "an"
    action_domain_file = "data/action_domain.json"
    chubot = ChuBotAction(botname, language='vi')
    chubot.load_domain(action_domain_file)
    chubot.run_commandline_bot()

def check_similarity(sentence,data,entity):
    # print("sent entity : ",entity)
    # print("data entity : ",data.get('entity'))
    count = 0 
    N = []
    NP = []
    V = []
    for token in nlp(sentence):
        if token.tag_=='Np' or token.tag_=='P':
            NP.append(token.text)
        if token.tag_=='V':
            V.append(token.text)
    for n in NP:
        for n2 in data.get('tags').get('NP'):
            if n == n2 : 
                count = count + 1
    for n in V:
        for n2 in data.get('tags').get('V'):
            if n == n2 : 
                count = count + 1
    for ent in entity:
        for ent2 in data.get('entity'):
            if ent == ent2:
                count = count + 1
    return count

def test_predict(ectracted_questions,answers):
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
        # print(action.chubot.nlp(inmessage))
        
        # print(responses)
        print("\n")
        print(entities)
        (prob, intent) = responses[0]
        # print(prob)
        print(intent)
        response_line = ''
        if intent != 'chat' and intent !='greeting' and intent!='introduce' and intent!='introduce_vnu' and intent !='command_lead_way' and intent !='command_return':
            max = 0
            
            for question in ectracted_questions:
                parts = question['question'].split(',')
                intent_quest = parts[0]
                # quest=parts[2]
                if intent==intent_quest:
                # print(max)
                    if check_similarity(inmessage,question,entities) > max :
                        max = check_similarity(inmessage,question,entities)
                        if max > 2:
                            response_line = answers[ectracted_questions.index(question)] + ' '+str(ectracted_questions.index(question))
        
        
        command_code = 0
        mp3 = -1
        for command in list_command_code:
            if(command.get("intent") == intent):
                print(command.get("command_code"))
                command_code = command.get("command_code")
        response = action.handle_message(inmessage)
        if response_line!='':
            response = response_line
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
    nlp = spacy.load('vi_spacy_model')
    # create_model('an')
    data = get_data()
    botname= 'an'
    action = ChuBotAction(botname)
    
    # app = Flask(__name__)
    questions = data.get('question')
    extracted_question = []
    for question in questions:
        entities = action.chubot.predict_entity(question)
        # print('entity',entities)
        N = []
        NP = []
        V = []
        for token in nlp(question):
            # if token.tag_=='N':
            #     N.append(token.text)
            if token.tag_=='Np' or token.tag_=='P':
                NP.append(token.text)
            if token.tag_=='V':
                V.append(token.text)
        extracted_question.append({'question':question,'entity':entities,'tags':{'N':N,'NP':NP,'V':V}})
    print('answer num',len(data.get('answer')))
    print('quest num',len(data.get('question')))
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

   
    test_predict(extracted_question,data.get('answer'))
    # test_predict()
    # # test_response()
    # bot = ChatBotAPI('vi', 'an')
    # bot.load_model()
    
    # line = bot.predict_message("who are you")
    # print(line)
    # print(detect("War doesn't show who's right, just who's left."))
    # detect("War doesn't show who's right, just who's left.")
