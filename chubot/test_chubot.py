from chubot_brain import ChuBotBrain
from chubot_action import ChuBotAction
import sys

def test_nercrf():
    nludatafile = "data/nlu_greet.json"

    chubot = ChuBotBrain("greet_en", language='en')
    chubot.load_data(nludatafile)

    meta = chubot.train_nercrf()
    print(meta)

    inmessage = "a bird and a dog"

    tagged_entities = chubot.predict_entity(inmessage)

    print(tagged_entities)


def test_intent_classification():
    nludatafile = "data/nlu_greet.json"

    chubot = ChuBotBrain("greet_en", language='en')
    chubot.load_data(nludatafile)
    meta = chubot.train_intent_classification()
    print(meta)

    inmessage = "sad but great. want a dog picture"
    intent_probs = chubot.predict_intent(inmessage)
    print(intent_probs)


def test_nlu_predict():
    nludatafile = "data/nlu_greet.json"

    chubot = ChuBotBrain("greet_en", language='en')

    inmessage = "a bird and a dog"
    tagged_entities = chubot.predict_entity(inmessage)
    print(inmessage)
    print(tagged_entities)

    inmessage = "sad but great. want a dog picture"
    intent_probs = chubot.predict_intent(inmessage)
    print(inmessage)
    print(intent_probs)

def test_train_models():
    botname = "greet_en"
    nludatafile = "data/nlu_greet.json"

    chubot = ChuBotBrain(botname, language='en')
    chubot.load_data(nludatafile)
    meta = chubot.train()
    print(meta)

def test_response():
    botname = "greet_en"
    action_domain_file = "data/action_domain.json"
    chubot = ChuBotAction(botname, language='en')
    chubot.load_domain(action_domain_file)
    chubot.run_commandline_bot()


if __name__ == '__main__':
    # test_nercrf()
    # test_intent_classification()
    #
    # test_train_models()
    # test_nlu_predict()

    test_response()
