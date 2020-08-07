from api import ChatBotAPI
from flask import Flask
from flask import request
import datetime
if __name__=="__main__":
    domain_file = "/media/nvidia/ssd/catkin_ws/src/chu_bot_source/chubot/usingdata/new_domain.json"
    api = ChatBotAPI('vi','an')
    print("load answers")
    api.load_answers()
    print("load model chatbot")
    api.load_domain(domain_file)
    api.load_model()
    api.upload_answer()
    app = Flask(__name__)
    @app.route('/')
    def hello_world():
        if request.method == 'GET':
            old_time = datetime.datetime.now()
            mess = request.args.get('mess', '')
            print(mess)
            # line = predict(mess)
            line = api.get_answer(mess)
            # print(str(line))
            new_time = datetime.datetime.now()
            print("chatbot thinking time : ",new_time-old_time)
            return str(line).replace("'",'"')

    #     # return "null"

    app.run(host= '0.0.0.0')
    # while True:
    #     print("please type your input >>>>>>")
    #     text  = input()
    #     old_time = datetime.datetime.now()
    #     response = api.get_answer(text)
    #     new_time = datetime.datetime.now()
    #     print("status",api.is_in_session)
    #     print("timestamp",api.last_request_moment)
    #     print(response)
    #     print(new_time-old_time)