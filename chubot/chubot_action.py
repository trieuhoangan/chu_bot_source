from chubot_brain import ChuBotBrain
import json
import io
import os
import sys


class ChuBotAction():

    def __init__(self, botname, language='vi'):
        self.botname = botname
        self.chubot = ChuBotBrain(botname, language)
        self.followup_actions = None
        self.action_templates = None
        self.action_custom = None
        self.slots = None
        self.start_flag = True
        self.active_entities = {}
        # TODO perhaps if folder exists -> load metadata file

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

    def begin_conversation(self, **kwargs):
        """Say welcome
        """
        # TODO should fix name of this action or not?
        start_flag = False
        return self.handle_action("bot_welcome")

    def handle_message(self, inmessage, debug=False, intent_prob_threshold=0.3, **kwargs):
        """Simple rule_based response
        """
        import itertools

        # TODO handle first welcome, repetitive input, etc -> state of conversation
        entities_pred = self.chubot.predict_entity(inmessage)
        intents_pred = self.chubot.predict_intent(inmessage)
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

        bot_responses = [self.handle_action(
            action, **active_entities) for action in bot_actions]
        bot_responses = list(itertools.chain(*bot_responses))

        return bot_responses

    def run_commandline_bot(self, **kwargs):
        """A commandline bot
        """
        # TODO handle action_domain_file doesnt match entities and intent in nludata file

        print("Type stop to end.\n -------------------")
        begin = self.begin_conversation()
        print("Chubot:", begin)
        while True:
            sys.stdout.write("You   > ")
            inmessage = input()
            if inmessage == 'stop':
                break
            responses = self.handle_message(inmessage, debug=False)
            for response in responses:
                print("Chubot:", response)

    def custom_handle_message(self, inmessage, debug=False, intent_prob_threshold=0.3, **kwargs):
        """Simple rule_based response
        """
        import itertools

        # TODO handle first welcome, repetitive input, etc -> state of conversation
        entities_pred = self.chubot.predict_entity(inmessage)
        intents_pred = self.chubot.predict_intent(inmessage)
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

        bot_responses = [self.handle_action(
            action, **active_entities) for action in bot_actions]
        bot_responses = list(itertools.chain(*bot_responses))

        return bot_responses
