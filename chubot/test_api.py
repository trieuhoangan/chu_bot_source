from api import ChatBotAPI
if __name__=="__main__":
    domain_file = "usingdata/new_domain.json"
    api = ChatBotAPI('vi','an')
    api.load_answers()
    api.load_domain(domain_file)
    api.load_model()
    text  = input()
    response = api.predict_message(text)
    print(response)