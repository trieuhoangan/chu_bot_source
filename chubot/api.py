import json
from chubot_brain import ChuBotBrain
import sklearn_crfsuite
import io
import csv
from answer_retrieval import ChitChat


class ChatBotAPI():
    def __init__(self, language, botname):
        self.followup_actions = []
        self.action_templates = []
        self.action_custom = []
        self.slots = []
        domain_file = "data/new_domain.json"
        # self.load_domain(action_domain_file)
        self.chatbot = ChuBotBrain(botname, "vi")
        self.tfidf = None
        self.le = None
        self.crf = None
        self.clf = None
        with io.open(domain_file) as f:
            action_domain = json.loads(
                open(domain_file, encoding="utf-8").read())

        # self.followup_actions = None
        # self.action_templates = None
        # self.action_custom = None
        # self.slots = None
        self.followup_actions = action_domain["action_domain_data"]["followup_actions"]
        self.action_templates = action_domain["action_domain_data"]["action_templates"]
        self.action_custom = action_domain["action_domain_data"]["action_custom"]
        self.slots = action_domain["action_domain_data"]["slots"]

    def load_domain(self, domain_file):
        """Load domain info to bot
        """
        # TODO handle exception here
        # setter -hmm, is this okay? should use @property or not?
        with io.open(domain_file) as f:
            action_domain = json.loads(
                open(domain_file, encoding="utf-8").read())

        # data fields
        self.followup_actions = action_domain["action_domain_data"]["followup_actions"]
        self.action_templates = action_domain["action_domain_data"]["action_templates"]
        self.action_custom = action_domain["action_domain_data"]["action_custom"]
        self.slots = action_domain["action_domain_data"]["slots"]

    def load_model(self):
        import cloudpickle
        from sklearn.externals import joblib
        crf = joblib.load(self.chatbot.crf_model_path)
        self.crf = crf
        with io.open(self.chatbot.intent_cls_model_path, "rb") as f:
            tfidf, le, clf = cloudpickle.load(f)
        self.clf = clf
        self.le = le
        self.tfidf = tfidf

    def handle_action(self, action, **kwargs):
        """Handle a response action
            Select a random template for the action
            or fill the slot
            return a list of responses based on action
        """
        import random
        import re
        # TODO Handle KeyError dictionary
        if action in self.action_templates:
            # if action is a template
            templates = self.action_templates.get(action)
            template = random.choice(templates)

            # TODO handle multiple detected entity but template only one?
            # ->should in custom action
            # TODO handle multiple slots
            regex_pattern = r"{{(.*)}}"
            template_entity = re.search(regex_pattern, template, re.M | re.I)
            if not template_entity:
                # no found
                # return template as response
                return [template]
            else:
                active_entities = kwargs['active_entities']
                # TODO multiple slot filling
                entity = template_entity.group(1)
                for ent in active_entities:
                    if ent['entity'] == entity:
                        value = ent['value']
                return [template.replace(template_entity.group(), value)]

        elif action in self.action_custom:
            import custom_actions
            func = getattr(
                custom_actions, self.action_custom[action].get("func_name"))
            response = func(
                action_args=self.action_custom[action].get("kwargs"), **kwargs)

        return response

    def handle_message(self, inmessage, debug=False, intent_prob_threshold=0.3, **kwargs):
        import itertools

        # TODO handle first welcome, repetitive input, etc -> state of conversation
        entities_pred = self.predict_entity(inmessage)
        intents_pred = self.predict_intent(inmessage)
        if debug:
            print("Predicted entity: ", entities_pred)
            print("Predicted intent: ", intents_pred)

        # select the first ranked predicted entity/intents
        # handle entities with different values but same entity type

        active_entities = {"active_entities": entities_pred}
        # select highest probable intent
        (prob, intent) = intents_pred[0]

        if prob < intent_prob_threshold:
            bot_actions = ["default_fallback"]
        else:
            # TODO handle key error?
            bot_actions = self.followup_actions.get(intent)

        if debug:
            print(bot_actions)

        bot_responses = [self.handle_action(action, **active_entities) for action in bot_actions]
        bot_responses = list(itertools.chain(*bot_responses))

        return bot_responses

    def predict_intent(self, message):
        inmessage_tokens = [token.text for token in self.chatbot.nlp(message)]
        inmessage_join_tokens = " ".join(inmessage_tokens)
        inmessage_vector = self.tfidf.transform([inmessage_join_tokens])
        # predict the probabilies
        y_probs = self.clf.predict_proba(inmessage_vector)
        
        # labels of each classes
        labels = self.le.inverse_transform(self.clf.classes_)
        # print(labels)
        y_probs_with_labels = list(zip(y_probs[0], labels))
        # sort the probabilities retrieving the indices of
        # the elements in sorted order
        # TODO, try this method: sorted_indices = np.fliplr(np.argsort(pred_result, axis=1))
        y_probs_with_labels.sort(key=lambda v: -v[0])

        return y_probs_with_labels

    def predict_entity(self, message):
        indoc = self.chatbot.nlp(message)
        insent = [(ii.text, ii.tag_, 'N/A') for ii in indoc]
        insent_features = self.chatbot.sent2features(insent)
        ent_probs = self.crf.predict_marginals_single(insent_features)
        # max of probability of each token
        ent_probs_sorted = [sorted(eprob.items(), key=lambda kv:(
            kv[1], kv[0]))[-1] for eprob in ent_probs]
        # TODO NOTE must handle entity with multiple tokens --> B,L, I tags not only U
        # TODO should change bilou to bio tag?

        ent_probs_idx = [(i, label, conf) for i, (label, conf)
                         in enumerate(ent_probs_sorted) if label != "O"]

        # also get the default value of the entity based on the entity_synonyms
        # TODO handle synonyms here!
        tagged_entities = [
            {'start': indoc[ix].idx,
             'end': indoc[ix].idx + len(indoc[ix].text),
             'value': indoc[ix].text,
             'entity': label[2:],
             'confidence': conf} for ix, label, conf in ent_probs_idx]

        return tagged_entities

    def predict_message(self, inmessage):
        intent_file = open('data/intent.csv', 'r', encoding="utf8")
        csvreader = csv.reader(intent_file, delimiter=',')
        list_command_code = []
        for row in csvreader:
            list_command_code.append(row)
        print(list_command_code)
        response = self.handle_message(inmessage)
        intents = self.predict_intent(inmessage)
        entities = self.predict_entity(inmessage)
        (prob, intent) = intents[0]
        command_code = 0
        mp3 = -1
        intent_id = 0
        chitchat_file = 'data/chitchat.csv'
        ask_what_file = 'data/ask_what.csv'
        ask_who_file = 'data/ask_who.csv'
        ask_where_file = 'data/ask_where.csv'
        ask_number_file = 'data/ask_number.csv'
        ask_when_file = 'data/ask_when.csv'
        chitchat = ChitChat(chitchat_file)
        ask_what = ChitChat(ask_what_file)
        ask_who = ChitChat(ask_who_file)
        ask_where = ChitChat(ask_where_file)
        ask_when = ChitChat(ask_when_file)
        ask_number = ChitChat(ask_number_file)
        for command in list_command_code:
            if(command[1] == intent):
                print(command[2])
                command_code = command[2]
                intent_id = command[0]
        if intent=='ask_where' and len(entities)==0:
            mp3 = 15
        if intent=='ask_number':
            most_similar_question, answer = ask_number.retrieve_answer(inmessage)
            print(most_similar_question)
            response = answer
        if intent=='ask_when':
            most_similar_question, answer = ask_when.retrieve_answer(inmessage)
            print(most_similar_question)
            response = answer
        if intent=='ask_who':
            most_similar_question, answer = ask_who.retrieve_answer(inmessage)
            print(most_similar_question)
            response = answer
        if intent=='ask_where':
            most_similar_question, answer = ask_where.retrieve_answer(inmessage)
            print(most_similar_question)
            response = answer
        if intent=='ask_what':
            most_similar_question, answer = ask_what.retrieve_answer(inmessage)
            print(most_similar_question)
            response = answer
        if intent=='chitchat':
            most_similar_question, answer = chitchat.retrieve_answer(inmessage)
            print(most_similar_question)
            response = answer
        if intent=='introduce_vnu':
            mp3 = int(response[0])
        result_json = {"intent": intent_id, "entities": entities,
                       "command_code": command_code, "response": response,'mp3':mp3}
        return json.dumps(result_json, ensure_ascii=False)
