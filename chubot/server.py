from chubot_action import ChuBotAction
from flask import Flask
import json
app = Flask(__name__)
import csv
from flask import request 
@app.route("/")
def home():
    return("hello")

class AnAction(ChuBotAction):
    pass

@app.route("/send_text",methods={'POST'})
def handle_message():
    inmessage = request.request.get_data()
    inmessage = inmessage.decode('utf-8')
    botname = 'an'
    bot  = ChuBotAction(botname)
    bot.load_domain("data/new_domain.json")
    intent_file = open('/data/intent.csv','r',encoding="utf8")
    csvreader = csv.reader(intent_file,delimiter=',')
    list_command_code=[]
    for row in csvreader:
        list_command_code.append(row)
    
    response = bot.handle_message(inmessage)
    intents = bot.chubot.predict_intent(inmessage)
    entities = bot.chubot.predict_entity(inmessage)
    (prob,intent) = intents[0]
    command_code = 0
    for command in list_command_code:
        if(command.get("intent")==intent):
            print(command.get("command_code"))
            command_code=command.get("command_code")
    result_json = {"intent":intent,"entities":entities,"command_code":command_code,"response":response}
    return json.dumps(result_json,ensure_ascii=False)


if __name__ == "__main__":
    
    app.run()

