from chubot_brain import ChuBotBrain
from chubot_action import ChuBotAction
import sys
import csv
import json
def craete_model(name):
    chubot = ChuBotBrain(name,language='vi')
    chubot.load_data("data/test.json")
    # chubot.load_data("data/vi_nlu_ask_way.json")
    meta = chubot.train()
    print(meta)
    print(chubot.entity_synonyms)
def test_response():
    botname = "an"
    action_domain_file = "data/action_domain.json"
    chubot = ChuBotAction(botname, language='vi')
    chubot.load_domain(action_domain_file)
    chubot.run_commandline_bot()
def test_predict():
    botname="an"
    action_domain_file = "data/new_domain.json"
    # chubot = ChuBotBrain(botname, language='vi')
    action = ChuBotAction(botname)
    action.load_domain(action_domain_file)
    intent_file = open('data/intent.csv','r',encoding="utf8")
    csvreader = csv.reader(intent_file,delimiter=',')
    list_command_code=[]
    for row in csvreader:
        list_command_code.append({"intent":row[1],"command_code":row[2]})

    while True:
        print("you >")
        inmessage = input()
        if inmessage == 'stop':
            break
        inmessage = inmessage.lower()
        responses= action.chubot.predict_intent(inmessage)
        entities = action.chubot.predict_entity(inmessage)
        print(action.chubot.nlp(inmessage))
        print(responses)
        print("\n")
        print(entities)
        (prob,intent) = responses[0]
        print(prob)
        print(intent)
        command_code = 0
        for command in list_command_code:
            if(command.get("intent")==intent):
                print(command.get("command_code"))
                command_code=command.get("command_code")
        response = action.handle_message(inmessage)
        result_json = {"intent":intent,"entities":entities,"command_code":command_code,"response":response}
        print(json.dumps(result_json,ensure_ascii=False))
        # print(result_json)

if __name__=="__main__":
    craete_model('an')
    # test_response()
    test_predict()
