# import pandas as pd
import json
import csv
import spacy



# for entity in  json_object.get("nlu_data").get("entity_synonyms"):
#     new_synonyms = []
#     for synonym in entity.get("synonyms"):
#         synonym_token = [token.text for token in nlp(synonym)]
#         new_synonyms.append(" ".join(synonym_token))
#     entity.__setitem__("synonyms",new_synonyms)
json_object = {"nlu_data": {"common_examples": [],
                                "regex_features": [], "lookup_tables": [], "entity_synonyms": []}}

def load_entity_data(entity_link):
    entity_file = open(entity_link, newline='', encoding="utf8")
    csv_reader = csv.reader(entity_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            symnonyms = row[2].split(',')
            json_object.get("nlu_data").get("entity_synonyms").append(
                {"entity": row[0], "original_value": row[1], "synonyms": symnonyms})

        line_count = line_count+1
    entity_file.close()

def load_quest_data(quest_link,text_col,intent_col):
    intent_file = open(quest_link, newline='', encoding="utf8")

    # nlp = spacy.load('vi_core_news_md')
    all_entity = json_object.get("nlu_data").get("entity_synonyms")
    csv_reader = csv.reader(intent_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count = line_count+1
            continue
        text = row[text_col].lower()
        intent = row[intent_col].lower()
        if text != "":
            entities = []

            for entity in all_entity:
                for symnonym in entity.get("synonyms"):
                    if symnonym in text:
                        start = text.index(symnonym)
                        if start > 1 and text[start-1] != ' ':
                            continue

                        end = start + len(symnonym) - 1
                        if end+1 != len(text):
                            if text[end+1] != ' ':
                                continue
                        entities.append({"start": start, "end": end, "entity": entity.get(
                            "entity"), "value": entity.get("original_value")})
            if len(entities) == 0:
                json_object.get("nlu_data").get("common_examples").append(
                    {"text": text, "intent": intent})
            else:
                json_object.get("nlu_data").get("common_examples").append(
                    {"text": text, "intent": intent, "entities": entities})
    intent_file.close()
def load_distinc_data(filename,intent_col,entity_col):
    intent_file = open(filename, newline='', encoding="utf8")

    # nlp = spacy.load('vi_core_news_md')
    all_entity = json_object.get("nlu_data").get("entity_synonyms")
    csv_reader = csv.reader(intent_file, delimiter=',')
    lines = []
    examples = []
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count = line_count+1
            continue
        text = row[intent_col].lower()
        intent = row[entity_col].lower()
        if text !='':
            if text not in lines:
                lines.append(text)
                examples.append({'text':text,'intent':intent})
    for example in examples:
        entities = []
        for entity in all_entity:
            for symnonym in entity.get("synonyms"):
                if symnonym in example.get('text'):
                    start = example.get('text').index(symnonym)
                    if start > 1 and example.get('text')[start-1] != ' ':
                        continue

                    end = start + len(symnonym) - 1
                    if end+1 != len(example.get('text')):
                        if example.get('text')[end+1] != ' ':
                            continue
                    entities.append({"start": start, "end": end, "entity": entity.get(
                        "entity"), "value": entity.get("original_value")})
        if len(entities) == 0:
            json_object.get("nlu_data").get("common_examples").append(
                {"text": example.get('text'), "intent": example.get('intent')})
        else:
            json_object.get("nlu_data").get("common_examples").append(
                {"text": example.get('text'), "intent": example.get('intent'), "entities": entities})
       
    intent_file.close()
if __name__=="__main__":
    load_entity_data('entity_list.csv')
    load_distinc_data('quest_data_noise.csv',2,0)
    load_quest_data('Q.csv',2,0)
    with open('test.json', 'w', encoding='utf8') as output:
        output.write(json.dumps(json_object, ensure_ascii=False))
        print('done')
