# -*- coding: utf-8 -*-
import json
import os
import spacy
import sklearn_crfsuite
import random
import io
import codecs

class ChuBotBrain():
    # TODO maybe init models folder here instead of handling in train*
    def __init__(self, name, language='vi'):
        
        self.name = name
        self.model_folder = "./models"
        
        self.crf_model_path = os.path.join(
            "./models", self.name + "_NERCRF.pkl")
        self.intent_cls_model_path = os.path.join(
            "./models", self.name + "_intent_classification.pkl")
        self.language = language
        self.common_examples = []
        self.regex_features = []
        self.entity_synonyms = []
        self.lookup_tables = []

        # mkdir if not existing
        # TODO handle existing folder
        try:
            os.makedirs(self.model_folder)
        except OSError:
            print("Creation of the directory %s failed (/Folder exists)" %
                  self.model_folder)
        else:
            print("Successfully created the directory %s " % self.model_folder)
        # load spacy nlp for pos tag
        # TODO handle loading exception
        if language == "en":
            self.nlp = spacy.load('en_core_web_sm')
        elif language == "vi":
            self.nlp = spacy.load('vi_spacy_model')

    def load_data(self, datafile, **kwargs):
        # TODO handle exception
        with open(datafile,'r',encoding='utf-8') as f:
            # print("xin chào")
            # print(f.read())

            data = json.loads(f.read())
        # print(data)
        # data fields
        for cell in data['nlu_data']['common_examples']:
            # if cell.get("text") == None:
            #     continue
            # print(cell)
            self.common_examples.append(cell)

        for cell in data['nlu_data']['entity_synonyms']:
            self.entity_synonyms.append(cell)

        for cell in data['nlu_data']['regex_features']:
            self.regex_features.append(cell)

        for cell in data['nlu_data']['lookup_tables']:
            self.lookup_tables.append(cell)

    def train(self, **kwargs):
        """Training
            train nercrf and intent classification models
        """
        crf_meta = self.train_nercrf(**kwargs)
        clf_meta = self.train_intent_classification(**kwargs)

        # ** implies that the argument is a dictionary.
        # Using ** [double star] is a shortcut that allows you to pass multiple arguments to a function directly using a dictionary.
        return {**crf_meta, **clf_meta}

    def train_nercrf(self, train_test_ratio=0.8, **kwargs):
        """
        *kwargs for future arguments of CRF and learning algorithm
        train_test_ratio= 1 if using all data for training
        TODO: handle multiple datafiles?
        """
        dataset = self.create_dataset()

        # TODO should add to kwargs or not
        if train_test_ratio < 1.0:
            random.shuffle(dataset)
            train_sents = dataset[:int(len(dataset)*train_test_ratio)]
            test_sents = dataset[int(len(dataset)*train_test_ratio):]
            x_train = [self.sent2features(s) for s in train_sents]
            y_train = [self.sent2labels(s) for s in train_sents]

            x_test = [self.sent2features(s) for s in test_sents]
            y_test = [self.sent2labels(s) for s in test_sents]

        elif train_test_ratio == 1.0:
            train_sents = dataset
            x_train = [self.sent2features(s) for s in train_sents]
            y_train = [self.sent2labels(s) for s in train_sents]

        # TODO handle invalid train_test_ratio (<=0)

        # training
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=50,
            #         all_possible_transitions=False,
            verbose=0
        )
        crf.fit(x_train, y_train)

        # TODO add to log? or using debug flag
        print("train accuracy", crf.score(x_train, y_train))
        if train_test_ratio < 1.0:
            print("test accuracy", crf.score(x_test, y_test))

        # save model
        from sklearn.externals import joblib

        joblib.dump(crf, self.crf_model_path)

        return {"path_to_crf_model": self.crf_model_path}

    def predict_entity(self, inmessage):
        """
        load nercrf model then predict entity/entities in the input sentence
        return a list of recognized entites (dict)
        """
        from sklearn.externals import joblib

        # TODO Handle load model exception
        crf = joblib.load(self.crf_model_path)

        # vectorize the inmessage
        indoc = self.nlp(inmessage)
        insent = [(ii.text, ii.tag_, 'N/A') for ii in indoc]
        insent_features = self.sent2features(insent)
        ent_probs = crf.predict_marginals_single(insent_features)
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

    def train_intent_classification(self, **kwargs):
        """TODO: split training data (or not? -> perhaps not)
        """

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import LabelEncoder
        # import xgboost
        from sklearn.svm import SVC
        import cloudpickle

        # tokenize using spacy
        x_docs = [self.nlp(example['text'])
                  for example in self.common_examples]
        y_labels = [example['intent'] for example in self.common_examples]

        # extract tokens from Spacy Doc objects
        x_tokenized = [[token.text for token in doc] for doc in x_docs]
        # join tokens
        x_join_tokens = [" ".join(token_list) for token_list in x_tokenized]

        # vectorize x
        # TODO Word embedding
        tfidf = TfidfVectorizer()
        x_train = tfidf.fit_transform(x_join_tokens)
        # print(x_train[0].todense())   ## x_train is in compressed sparse row format
        # transform y
        le = LabelEncoder()
        y_train = le.fit_transform(y_labels)

        # classifier
        # TODO do a gridsearch for xgboost or just a simple linear svc?
        clf = SVC(kernel='linear', C=1, probability=True)
        # clf = xgboost.XGBClassifier(max_depth=7,
        #                    min_child_weight=1,
        #                    learning_rate=0.1,
        #                    n_estimators=500,
        #                    objective='binary:logistic',
        #                    gamma=0,
        #                    max_delta_step=0,
        #                    subsample=1,
        #                    colsample_bytree=1,
        #                    colsample_bylevel=1,
        #                    reg_alpha=0,
        #                    reg_lambda=0,
        #                    scale_pos_weight=1,
        #                    seed=1,
        #                    missing=None,
        #                    verbosity=3)
        clf.fit(x_train, y_train)

        # save model
        # 1. tfidf model
        # 2. labels of intents from LabelEncoder
        # 3. classification model
        models = (tfidf, le, clf)
        with io.open(self.intent_cls_model_path, "wb") as f:
            cloudpickle.dump(models, f)

        return {"path_to_clf_model": self.intent_cls_model_path}

    def predict_intent(self, inmessage):
        """inmessage: Text -> sorted List[Tuple(probability: Float, intent: Text)]
        """
        import cloudpickle

        # load models
        # TODO handle exception?
        with io.open(self.intent_cls_model_path, "rb") as f:
            tfidf, le, clf = cloudpickle.load(f)

        # Tokenize message using spacy nlp
        # Vectorize using tfidf
        inmessage_tokens = [token.text for token in self.nlp(inmessage)]
        inmessage_join_tokens = " ".join(inmessage_tokens)
        inmessage_vector = tfidf.transform([inmessage_join_tokens])
        # predict the probabilies
        y_probs = clf.predict_proba(inmessage_vector)

        # labels of each classes
        labels = le.inverse_transform(clf.classes_)
        # print(labels)
        y_probs_with_labels = list(zip(y_probs[0], labels))
        # sort the probabilities retrieving the indices of
        # the elements in sorted order
        # TODO, try this method: sorted_indices = np.fliplr(np.argsort(pred_result, axis=1))
        y_probs_with_labels.sort(key=lambda v: -v[0])

        return y_probs_with_labels

    @staticmethod
    def load_json_data(datafile):
        # deprecated!!!
        # TODO handle exception here
        with open(datafile, 'r',encoding="utf-8") as f:
            data = json.loads(f.read())

        # data fields
        common_examples = data['nlu_data']['common_examples']
        entity_synonyms = data['nlu_data']['entity_synonyms']
        regex_features = data['nlu_data']['regex_features']
        lookup_tables = data['nlu_data']['lookup_tables']

        # reformat entity_synonyms
        return (common_examples, entity_synonyms, regex_features, lookup_tables)

    def create_dataset(self, **kwargs):
        """dataset is a list of List[List[Tuple[token, POS-tag, entity]]] for NERCRF
        """

        # dataset
        dataset = []
        with open('log.txt','w',encoding='utf-8') as f:
            f.write(str(self.common_examples))
        
        for example in self.common_examples:
            #  List[(start_index, end_index, entity)]
            # if example.get('text')==None:
            #     continue
            entity_offsets = [(e["start"], e["end"], e["entity"])
                              for e in example.get("entities", [])
                              ]

            # tokenize each example and get pos_tag
            # print(example["text"])

            doc = self.nlp(example["text"])

            # bilou tags :: a list of tags
            # for example
            # tokens :: [show, me, a, mexican, place, in, the, centre]
            # BILOU  :: [O,    O,  O,'U-cuisine', O,   O, O, 'U-location']
            bilou_tags = spacy.gold.biluo_tags_from_offsets(
                doc, entity_offsets)
            example_data = [(token.text, token.tag_, bilou_tags[i])
                            for i, token in enumerate(doc)]
            dataset.append(example_data)

        return dataset

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
