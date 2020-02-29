from api import ChatBotAPI
if __name__=="__main__":
    domain_file = "usingdata/new_domain.json"
    api = ChatBotAPI('vi','an')
    api.load_answers()
    api.load_domain(domain_file)
    api.load_model()
    while True:
        text  = input()
        response = api.get_answer(text)
        print("status",api.is_in_session)
        print("timestamp",api.last_request_moment)
        print(response)