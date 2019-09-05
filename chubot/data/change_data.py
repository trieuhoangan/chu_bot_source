# import pandas as pd
import json
import csv
import spacy

intent_file = open('quest_data_noise.csv', newline='', encoding="utf8")
entity_file = open('entity_list.csv', newline='', encoding="utf8")
json_object = {"nlu_data": {"common_examples": [],
                            "regex_features": [], "lookup_tables": [], "entity_synonyms": []}}
# nlp = spacy.load('vi_core_news_md')

csv_reader = csv.reader(entity_file, delimiter=',')
line_count = 0
for row in csv_reader:
    if line_count != 0:
        symnonyms = row[2].split(',')
        json_object.get("nlu_data").get("entity_synonyms").append(
            {"entity": row[0], "original_value": row[1], "synonyms": symnonyms})

    line_count = line_count+1

entity_file.close()
all_entity = json_object.get("nlu_data").get("entity_synonyms")

csv_reader = csv.reader(intent_file, delimiter=',')
line_count = 0
for row in csv_reader:
    if line_count == 0:
        line_count = line_count+1
        continue
    text = row[2].lower()
    intent = row[0].lower()
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


# for entity in  json_object.get("nlu_data").get("entity_synonyms"):
#     new_synonyms = []
#     for synonym in entity.get("synonyms"):
#         synonym_token = [token.text for token in nlp(synonym)]
#         new_synonyms.append(" ".join(synonym_token))
#     entity.__setitem__("synonyms",new_synonyms)

intent_file.close()

with open('test.json', 'w', encoding='utf8') as output:
    output.write(json.dumps(json_object, ensure_ascii=False))
    print('done')
