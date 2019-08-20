"""Define all custom actions here"""
import requests
from chubot_brain import ChuBotBrain

def retrieve_image_deplicated(action_args, **kwargs):

    #TODO make a synonym mapper class

    def get_default_synonym(entity, value, entity_ldict):
        for ent in entity_ldict:
            if entity == ent['entity']:
                if value in ent['synonyms']:
                    return ent['default_value']

    nludatafile = "data/nlu_greet.json"

    chubot = ChuBotBrain("greet_en", language='en')
    chubot.load_data(nludatafile)
    entity_synonyms = chubot.entity_synonyms

    #handle arguments
    arg_entity = action_args["entity"]
    arg_value_type = action_args["value"]

    active_entities = kwargs["active_entities"]

    for ent in active_entities:
        if ent['entity'] == arg_entity:
            if arg_value_type == "value":
                arg_value = ent['value']
            elif arg_value_type == "default_value":
                arg_value = get_default_synonym(ent['entity'], ent['value'], entity_synonyms)

    print(arg_value)
    # print('http://shibe.online/api/{}?count=1&urls=true&httpsUrls=true'.format(arg_value))
    r = requests.get('http://shibe.online/api/{}?count=1&urls=true&httpsUrls=true'.format(arg_value))

    response = r.content.decode()
    response = response.replace('["', '')
    response = response.replace('"]', '')

    return response


def retrieve_image(action_args, **kwargs):

    def get_image(animal):

        if animal not in ["birds", "cats", "shibes"]:
            return "I can't find that animal!"

        r = requests.get('http://shibe.online/api/{}?count=1&urls=true&httpsUrls=true'.format(animal))

        response = r.content.decode()
        response = response.replace('["', '')
        response = response.replace('"]', '')
        return response

    synmap = {
        "birds" : ["bird", "birds", "puffin"],
        "shibes" :["dog", "dogs", "puppy", "shibes"],
        "cats" : ["cat", "kitten", "cats"]
    }

    active_entities = kwargs['active_entities']
    request_args = []
    for ent in active_entities:
        for k,v in synmap.items():
            if ent['value'] in v:
                request_args.append(k)

    return [get_image(animal) for animal in request_args]
