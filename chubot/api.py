import numpy as np
import time
import json
from chubot_brain import ChuBotBrain
import sklearn_crfsuite
import io
import csv
from answer_retrieval import ChitChat
import os
import spacy
import datetime
from gensim.models import KeyedVectors
class ChatBotAPI():
    def __init__(self, language, botname):
        self.standard_prob = 0.0
        self.is_in_session = False
        self.followup_actions = []
        self.action_templates = []
        self.action_custom = []
        self.slots = []
        self.name = botname
        self.model_folder = "./models"
        self.crf_model_path = os.path.join(
            "./models", self.name + "_NERCRF.pkl")
        self.intent_cls_model_path = os.path.join(
            "./models", self.name + "_intent_classification.pkl")
        domain_file = "./usingdata/new_domain.json"
        # self.load_domain(action_domain_file)
        # self.chatbot = ChuBotBrain(botname, "vi")
        self.tfidf = None
        self.le = None
        self.crf = None
        self.clf = None
        self.lead_to_section_code =2
        self.go_around_code = 1
        self.speak_code = 0
        self.use_mp3_code = 3
        self.code = 0
        self.mp3 = -1
        self.section_id = -1
        self.last_request_moment = 0
        self.sessinon_length = 50
        with io.open(domain_file) as f:
            action_domain = json.loads(
                open(domain_file, encoding="utf-8").read())
        # word2vecmodel_link = "models/wiki.vi.model.bin"
        # self.word2vec = KeyedVectors.load_word2vec_format(fname=word2vecmodel_link,binary=True,unicode_errors='strict')
        # self.followup_actions = None
        # self.action_templates = None
        # self.action_custom = None
        # self.slots = None
        self.followup_actions = action_domain["action_domain_data"]["followup_actions"]
        self.action_templates = action_domain["action_domain_data"]["action_templates"]
        self.action_custom = action_domain["action_domain_data"]["action_custom"]
        self.slots = action_domain["action_domain_data"]["slots"]
    def load_answers(self):
        chitchat_file = 'usingdata/chitchat.csv'
        ask_what_file = 'usingdata/ask_what.csv'
        ask_who_file = 'usingdata/ask_who.csv'
        ask_where_file = 'usingdata/ask_where.csv'
        ask_number_file = 'usingdata/ask_number.csv'
        ask_when_file = 'usingdata/ask_when.csv'
        lead_to_section_file = 'usingdata/lead_to_section.csv'
        answer_retriever = ChitChat(chitchat_file)
        answer_retriever.add_more_data(ask_what_file)
        answer_retriever.add_more_data(ask_who_file)
        answer_retriever.add_more_data(ask_where_file)
        answer_retriever.add_more_data(ask_number_file)
        answer_retriever.add_more_data(ask_when_file)
        answer_retriever.add_more_data(lead_to_section_file)
        self.retriever = answer_retriever
    def upload_answer(self):
        self.retriever.print_all_data()
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
        # import joblib
        crf = joblib.load(self.crf_model_path)
        self.crf = crf
        with io.open(self.intent_cls_model_path, "rb") as f:
            tfidf, le, clf = cloudpickle.load(f)
        self.clf = clf
        self.le = le
        self.tfidf = tfidf
        self.nlp = spacy.load('vi_spacy_model')
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
        start = datetime.datetime.now()
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
        end = datetime.datetime.now()
        print("predict time: ",end-start)
        return [bot_responses,entities_pred,intents_pred]

    def predict_intent(self, message):  
        inmessage_tokens = [token.text for token in self.nlp(message)]
        inmessage_join_tokens = " ".join(inmessage_tokens)
        inmessage_vector = self.tfidf.transform([inmessage_join_tokens])

        # sent_vec = np.zeros(400)
        # for token in inmessage_tokens:
        #     if token in self.word2vec:
        #         vector = self.word2vec[token]
        #         sent_vec = np.add(sent_vec,vector)
        # if np.sqrt(sent_vec.dot(sent_vec)) != 0:
        #     sent_vec = sent_vec/np.sqrt(sent_vec.dot(sent_vec))
        # inmessage_vector = [sent_vec]
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
    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [label for token, postag, label in sent]

    def sent2tokens(self, sent):
        # token is the word!
        return [token for token, postag, label in sent]
    def predict_entity(self, message):
        indoc = self.nlp(message)
        insent = [(ii.text, ii.tag_, 'N/A') for ii in indoc]
        insent_features = self.sent2features(insent)
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
    def return_silent(self):
        result_json = {"mp3":-1,"section_id":-1,
            "code": 0, "response": [" "]}
        return result_json
    def return_unknown(self):
        result_json = {"mp3":-1,"section_id":-1,
            "code": 0, "response": ["xin lỗi bạn nói lại được không"]}
        return result_json   
    def new_flow(self,inmessage):
        if inmessage =="" or inmessage == " ":
            return self.return_silent()
        result = self.handle_message(inmessage)
        response = result[0]
        entities = result[1]
        intents  = result[2]
        (prob, intent) = intents[0]
        
        pass
    # the flow of old fuso, require user to say hello first before any other request. 
    # the turn the json form of message 
    def get_answer(self,inmessage):
        
        if inmessage =="" or inmessage == " ":
            return self.return_silent()
        # response = self.handle_message(inmessage)
        # intents = self.predict_intent(inmessage)
        # entities = self.predict_entity(inmessage)
        result = self.handle_message(inmessage)
        response = result[0]
        entities = result[1]
        intents  = result[2]
        (prob, intent) = intents[0]
        print(prob)
        if self.is_in_session == False:
            if intent == "greeting" and prob > self.standard_prob:
                self.is_in_session = True
                self.last_request_moment = time.time()
                return self.answer(entities,intent,prob,response,inmessage)
            else:
                return self.return_silent()
        else:
            timestamp = time.time()
            if (timestamp - self.last_request_moment) > self.sessinon_length:
                print("session expired")
                if intent == "greeting" and prob >self.standard_prob:
                    self.is_in_session = True
                    self.last_request_moment = time.time()
                    return self.answer(entities,intent,prob,response,inmessage)
                else:
                    self.is_in_session = False
                    return self.return_silent()
            else:
                self.last_request_moment = time.time()
                return self.answer(entities,intent,prob,response,inmessage)
    # return the json form of message and entity 
    def answer(self, entities,intent,prob,response,inmessage):
        print("intent",intent)
        start = datetime.datetime.now()
        mp3 = -1   
        section_id = -1
        code = 0
        # if prob < 0.5:
        #     return self.return_unknown()
        if str(response) == "100.mp3":
            mp3 = 100
            code = self.use_mp3_code
        if intent=='ask_where' and len(entities)==0:
            mp3 = 15
        if intent=='chitchat':
            most_similar_question, answer = self.retriever.retrieve_answer(inmessage,0)[0]
            print(most_similar_question)
            response = answer
        ## open mp3 file to introduce the room
        
        if intent=='ask_what':
            most_similar_question, answer = self.retriever.retrieve_answer(inmessage,1)[0]
            response = answer
            print(most_similar_question)
            if str(response) == "100.mp3":
                mp3 = 100
                code = self.use_mp3_code
            for entity in entities:
                if entity['entity'] == 'present' and entity['confidence'] > 0.85:
                    mp3 = 100
                    code = self.use_mp3_code
                    response = "100.mp3"
        if intent=='ask_who':
            most_similar_question, answer = self.retriever.retrieve_answer(inmessage,2)[0]
            response = answer
            print(most_similar_question)
        if intent=='ask_where':
            most_similar_question, answer = self.retriever.retrieve_answer(inmessage,3)[0]
            response = answer
            print(most_similar_question)
        if intent=='ask_number':
            most_similar_question, answer = self.retriever.retrieve_answer(inmessage,4)[0]
            response = answer
            print(most_similar_question)
        if intent=='ask_when':
            most_similar_question, answer = self.retriever.retrieve_answer(inmessage,5)[0]
            response = answer
            print(most_similar_question)
        if intent =='command_lead_way' and len(entities)!=0:
            for entity in entities:
                if entity["entity"] =="section":
                    most_similar_question, answer = self.retriever.retrieve_answer(inmessage,6)[0]
                    section_id = answer
                    code = self.lead_to_section_code
                if entity["entity"] =="area":
                    code = self.go_around_code
            
        if intent=='ask_where' and len(entities)==0:
            mp3 = 15
            code = self.use_mp3_code
        end = datetime.datetime.now()
        result_json = {"mp3":mp3,"section_id":section_id,
                "code": code, "response": response}
        print("answer time: ",end-start)
        print("response ",response)
        return result_json
    