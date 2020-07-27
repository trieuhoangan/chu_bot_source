from api import ChatBotAPI
from flask import Flask
from flask import request

if __name__=="__main__":
    domain_file = "usingdata/new_domain.json"
    api = ChatBotAPI('vi','an')
    api.load_answers()
    api.load_domain(domain_file)
    api.load_model()
    api.upload_answer()
    app = Flask(__name__)
    @app.route('/')
    def hello_world():
        if request.method == 'GET':
            mess = request.args.get('mess', '')
            print(mess)
            # line = predict(mess)
            line = api.get_answer(mess)
            # print(str(line))
            return str(line).replace("'",'"')

    #     # return "null"

    app.run(host= '0.0.0.0')
    # while True:
    #     text  = input()
    #     response = api.get_answer(text)
    #     print("status",api.is_in_session)
    #     print("timestamp",api.last_request_moment)
    #     print(response)