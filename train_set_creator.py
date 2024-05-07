# import needed libraries
import helper.Utils as Utils
import pandas as pd
import os
import configure as cf
import pyterrier as pt
import re
import logging
from mono_bert_test import MonoBertTester
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import helper.classifier as classifier
from snowballstemmer import stemmer
import arabicstopwords.arabicstopwords as ar_stp
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

class MonoBertTrainSetCreator():


    def __init__(self, qrels_path, vclaims_path, index_path, eval_metrics, bm25_search_depth=200, lang="en"):
        if not pt.started():
            print("Enabling PRF in pyterier")
            pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

        if qrels_path != "":
            self.df_qrels = pd.read_csv(qrels_path, sep="\t", names=[cf.QID, cf.Q0, cf.DOC_NO, cf.LABEL])
            self.df_qrels[cf.QID]=self.df_qrels[cf.QID].astype(str)
            self.df_qrels[cf.DOC_NO]= self.df_qrels[cf.DOC_NO].astype(str)
        else:
            self.df_qrels = None

        self.df_claim = Utils.read_file(vclaims_path)
        self.df_claim[cf.VCLAIM_ID]= self.df_claim[cf.VCLAIM_ID].astype(str)
        self.df_claim.set_index(cf.VCLAIM_ID, inplace=True)
        self.eval_metrics = eval_metrics
        self.ar_stemmer = stemmer("arabic")
        self.bm25_search_depth = bm25_search_depth
        self.porter= PorterStemmer()

        if not os.path.isfile(index_path): # if index is not built yet, build a new one
            self.index = self.build_multi_field_index(vclaims_path, index_path, lang)
        else:
            self.index = self.load_index(index_path)

    
    def load_index(self, index_path):
        try:
             # first load the index
            multi_field_index = pt.IndexFactory.of(index_path)
            # call getCollectionStatistics() to check the stats
            print(multi_field_index.getCollectionStatistics().toString())
            print("Index has been loaded successfully")
            return multi_field_index
        except Exception as e:
            print('Cannot load the index, check exception details {}'.format(e))
            return []


    def build_multi_field_index(self, claims_file, index_save_path, lang="en"):

        def get_document():
            for i, row in df_doc.iterrows():
                yield {cf.DOC_NO: row[cf.DOC_NO], cf.TEXT: row[cf.TEXT], cf.TITLE: row[cf.TITLE]}

        print("Creating index for vclaims collection within path ", claims_file, " and language is ", lang)
        preprocess_fn = self.preprocess_english
        iter_indexer = pt.IterDictIndexer(index_save_path,  overwrite=True, verbose=True)
        iter_indexer.setProperty("tokeniser", "EnglishTokeniser")

        if lang == "ar": 
            preprocess_fn = self.preprocess_arabic
            iter_indexer.setProperty("tokeniser", "UTFTokeniser")  # Replaces the default EnglishTokeniser, which makes assumptions specific to English
            iter_indexer.setProperty("termpipelines", "") # Removes the default PorterStemmer (English)


        # load the documents (verified claims) and apply preprocessing steps over them
        df_doc = Utils.read_file(claims_file)  
        df_doc[cf.VCLAIM] = df_doc[cf.VCLAIM].apply(preprocess_fn)
        df_doc[cf.TITLE] = df_doc[cf.TITLE].apply(preprocess_fn)
        df_doc[cf.TEXT] = df_doc[cf.VCLAIM].astype(str)
        df_doc[cf.DOC_NO] = df_doc[cf.VCLAIM_ID].astype(str)


        # the default is an English tokenizer: Tokenises text obtained from a text stream assuming English language.
        indexref = iter_indexer.index(get_document(), fields=[cf.TEXT, cf.TITLE], meta=[cf.DOC_NO])
        print(indexref.toString())

        # Load the index from a path
        multi_field_index = pt.IndexFactory.of(indexref)
        print("Index was build and saved within the path ", indexref.toString())
        print(multi_field_index.getCollectionStatistics().toString())

        return multi_field_index



    # make one row in trec format
    def get_one_row(self, tweet_id, tweet_text, vclaim_id, vclaim_text, label, rank, score, lexical_similarity=0, 
                    semantic_similarity=0, title=""):
        new_row = {
            cf.TWEET_ID: tweet_id,
            cf.TWEET_TEXT: tweet_text,
            cf.VCLAIM_ID: vclaim_id,
            cf.VCLAIM: vclaim_text,
            cf.LABEL: label,
            cf.RANK: rank,
            cf.SCORE: score,
            cf.LEXICAL_SIMILARITY: lexical_similarity,
            cf.SEMANTIC_SIMILARITY: semantic_similarity,
            cf.TITLE: title,
        }

        return new_row

    #removing stop sords function
    def ar_remove_stop_words(self, sentence):
        terms=[]
        stopWords= set(ar_stp.stopwords_list())
        for term in sentence.split() : 
            if term not in stopWords :
                terms.append(term)
        return " ".join(terms)


        #a function to normalize the tweets
    def normalize_arabic(self, text):
        text = re.sub("[إأٱآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)
        return(text)


    def ar_stem(self, sentence):
        return " ".join([self.ar_stemmer.stemWord(i) for i in sentence.split()])


        #removing stop sords function
    def en_remove_stop_words(self, sentence):
        terms=[]
        stop_words= set(stopwords.words('english'))
        words = sentence.split()
        for term in words: 
            if term not in stop_words :
                terms.append(term)
        return " ".join(terms)


    def en_stem(self, sentence):
        token_words=word_tokenize(sentence)
        return " ".join([self.porter.stem(word) for word in token_words])


    def preprocess_english(self, sentence):
        # apply preprocessing steps on the given sentence
        sentence = sentence.lower()
        sentence =self.en_remove_stop_words(sentence)
        sentence =self.en_stem(sentence)
        return sentence

    def preprocess_arabic(self, sentence): # for Arabic
        # apply preprocessing steps on the given sentence
        sentence = Utils.remove_emoji_smileys(sentence)
        sentence = self.normalize_arabic(sentence)
        sentence = self.ar_remove_stop_words(sentence)
        sentence = self.ar_stem(sentence)
        return sentence


    def clean_query_for_search(self, query_path, data_column="cleaned", lang="en"):

        print("Cleaning queries and applying preprocessing steps")
        df_query = Utils.read_file(query_path)
        # try test quries after extracting some information from urls
        df_query[cf.QUERY] =df_query[data_column].apply(Utils.clean)
        df_query[cf.QUERY] =df_query[cf.QUERY].apply(Utils.remove_emoji_smileys)
        df_query[cf.QUERY] =df_query[cf.QUERY].apply(Utils.remove_punctuation)

        if lang == "ar": # apply normalization, stemming and stop word removal for Arabic
            print("Applying normalization, stemming and stop word removal for Arabic")
            df_query[cf.QUERY] =df_query[cf.QUERY].apply(self.preprocess_arabic)
        else: # apply preprocessing for English
            print("Applying lowercasing, stemming and stop word removal for English")
            df_query[cf.QUERY] =df_query[cf.QUERY].apply(self.preprocess_english)
            

        df_query[cf.QID] = df_query[cf.TWEET_ID].astype(str)
        df_query = df_query[[cf.QID, cf.QUERY]]
        print("Done with cleaning!")
        return df_query
    


    def search_and_evaluate(self, query_path, run_save_path, retrieval_model="BM25", evaluation_path="",
                            data_column='cleaned', depth=100, method_name="BM25", DECIMAL_ROUND=5, lang="en"):
        
        df_query = self.clean_query_for_search(query_path, data_column=data_column, lang=lang)
        # intialize BM25 model to get the top 100 potentially relevant documents
        bm25_retr = pt.BatchRetrieve(self.index, controls = {"wmodel": retrieval_model},num_results=depth)
        print("Searching for the queries .....")
        # retrieve potentially relevant documents for each query in queries file
        bm25_res = bm25_retr.transform(df_query)
        # save the run in trec format
        bm25_res.to_csv(run_save_path, header=False, index=False, sep='\t')


        # Evaluate the performance
        bm25_eval = []
       
        # save evaluation results
        if evaluation_path != "":
            print("Perfomring evaluation ......")
            bm25_eval = pt.Utils.evaluate(bm25_res, self.df_qrels[["qid", "docno", cf.LABEL]],metrics=self.eval_metrics)
            bm25_eval.update({"name": method_name})
            bm25_eval.update({"depth": depth})
            bm25_eval = pd.DataFrame([bm25_eval])
            bm25_eval = bm25_eval.round(DECIMAL_ROUND)
            print(bm25_eval)


            if not os.path.isfile(evaluation_path): # if file is not exist, create a new one
                bm25_eval.to_excel(evaluation_path, index=False)
            else: # it is already exist, append current evaluation to it
                df_eval = Utils.read_file(evaluation_path)
                df_eval = df_eval.append(bm25_eval, ignore_index=True)
                df_eval.to_excel(evaluation_path, index=False)
               
        print("Done searching and evaluation ")
        return bm25_res, bm25_eval
    
    def create_test_pairs(self, query_path, run_path, query_column, pairs_save_path,):

        df_query = Utils.read_file(query_path)
        df_run = pd.read_csv(run_path, sep="\t", names=[cf.TWEET_ID, cf.DOCID, cf.DOC_NO, 
                                            cf.RANK, cf.SCORE, cf.QUERY],)
        df_result = pd.DataFrame(columns=[cf.TWEET_ID, cf.TWEET_TEXT, cf.VCLAIM_ID, 
                                        cf.VCLAIM, cf.LABEL, cf.RANK, cf.SCORE, cf.TITLE])

        df_query[cf.TWEET_ID] = df_query[cf.TWEET_ID].astype(str)
        df_run[cf.TWEET_ID] = df_run[cf.TWEET_ID].astype(str)
        df_run[cf.DOC_NO] = df_run[cf.DOC_NO].astype(str)
        
        df_run.set_index([cf.TWEET_ID, cf.DOC_NO], inplace=True)
        df_query.set_index(cf.TWEET_ID, inplace=True)

        l = 0 
        for ind, item in df_run.iterrows():
            tweet_id, vclaim_id = ind
            rank = df_run.loc[tweet_id, vclaim_id,][cf.RANK] + 1 # rank is zero-based, so we need to add 1 
            score = df_run.loc[tweet_id, vclaim_id,][cf.SCORE]
            # get the qrels rows for this query (tweet_id)

            tweet_text = df_query.at[tweet_id, query_column]
            vclaim_text = self.df_claim.at[vclaim_id, cf.VCLAIM]
            vclaim_title = self.df_claim.at[vclaim_id, cf.TITLE]
            
            if self.df_qrels is None: # if there is no qrels, assign zero as label for this row
                new_row = self.get_one_row(tweet_id, tweet_text, vclaim_id, vclaim_text, 
                                                cf.NEGATIVE_LABEL, rank, score, title=vclaim_title) 

            else:
                df_qrels_for_one_tweet = self.df_qrels[self.df_qrels[cf.QID] == tweet_id]
                # if vclaim_id is existed within qrels rows, then it is positive match, otherwise it is negative
                if vclaim_id in df_qrels_for_one_tweet[cf.DOC_NO].values:
                    new_row = self.get_one_row(tweet_id, tweet_text, vclaim_id, vclaim_text, 
                                                cf.POSITIVE_LABEL, rank, score, title=vclaim_title)  
                else:
                    new_row = self.get_one_row(tweet_id, tweet_text, vclaim_id, vclaim_text, 
                                                cf.NEGATIVE_LABEL, rank, score, title=vclaim_title) 

            df_result = df_result.append(new_row, ignore_index=True)
            l = l + 1
            if l % 1000 == 0:
                print(l, " rows have been created so far ")

        if not os.path.isfile(pairs_save_path): # create the directory if it does not exist
            os.makedirs(os.path.dirname(pairs_save_path), exist_ok=True)

        df_result.to_csv(pairs_save_path, index=False, sep='\t', encoding="utf-8")
        return df_result
    
        
    # for a given tweet_id, bring all positive examples from qrels file,
    # and add equal number of negative examples from the run
    def add_positive_and_negative(self, query_column, df_query, tweet_id, df_qrels_one, df_run, df_result, 
                                what_to_add=cf.VCLAIM_AND_TITLE, add_similarity=False, depth_of_random=20):
        
        df_one_query_run = df_run[df_run[cf.TWEET_ID] == tweet_id]
        df_vclaim_top_depth = df_one_query_run.head(depth_of_random)
        df_vclaim_top_depth = df_vclaim_top_depth.sample(frac=1) # shuffle the dataframe to choose random negative examples

        tweet_text = df_query.at[tweet_id, query_column]
        lexical_similarity = -1
        vclaim_semantic_similarity = -1
        title_semantic_similarity = -1
        
        target = len(df_qrels_one)
        if what_to_add == cf.VCLAIM_AND_TITLE:
            target = target * 2 # account for vclaim text and vclaim title
        df_one_query_run.set_index(cf.DOC_NO, inplace=True)

        # 1- Add positive examples from qrels file.
        # here df_qrels_one contains all rows that corresponds to one tweet only
        # so the tweet_id value will not change in the for loop
        for i, elem in df_qrels_one.iterrows():
            vclaim_id = elem[cf.DOC_NO]
            vclaim_text = self.df_claim.at[vclaim_id, cf.VCLAIM]
            vclaim_title = self.df_claim.at[vclaim_id, cf.TITLE]
            if vclaim_id in df_one_query_run.index: 
                rank = df_one_query_run.at[vclaim_id, cf.RANK] + 1 # rank is zero-based, so we need to add 1 
                score = df_one_query_run.at[vclaim_id, cf.SCORE]
                if add_similarity: #  if want to add lexical and semantic similarities
                    lexical_similarity = df_one_query_run.at[vclaim_id, cf.LEXICAL_SIMILARITY]
                    vclaim_semantic_similarity = df_one_query_run.at[vclaim_id, cf.SEMANTIC_SIMILARITY]
            else: # relevant document might not be within retrieved documents, so give it the least value
                rank = self.bm25_search_depth
                score = df_one_query_run[cf.SCORE].min()
                if add_similarity:
                    lexical_similarity = cf.EPSILON
                    vclaim_semantic_similarity = self.predict_similarity(tweet_text, vclaim_text)

            if add_similarity:
                title_semantic_similarity = self.predict_similarity(tweet_text, vclaim_title)

            vclaim_row = self.get_one_row(tweet_id, tweet_text, vclaim_id, vclaim_text, 
                                        cf.POSITIVE_LABEL, rank, score, lexical_similarity, vclaim_semantic_similarity)

            title_row = self.get_one_row(tweet_id, tweet_text, vclaim_id, vclaim_title, 
                                    cf.POSITIVE_LABEL, rank, score, lexical_similarity, title_semantic_similarity)       

            if what_to_add == cf.VCLAIM_ONLY:
                # Add the verified claim text
                df_result = df_result.append(vclaim_row, ignore_index=True)
            
            elif what_to_add == cf.TITLE_ONLY:
                # Add the verified claim title only                
                df_result = df_result.append(title_row, ignore_index=True)
            
            else:# what_to_add == cf.VCLAIM_AND_TITLE: 
                # Add the verified claim text and its title
                df_result = df_result.append(vclaim_row, ignore_index=True)
                df_result = df_result.append(title_row, ignore_index=True)

        # 2- Add negative examples by same number of positive examples
        # for the current tweet, go through the run and choose randomly from the top k vclaim
        # which are NOT included as positive examples, choose them as negative examples
        
        k = 0
        for i, elem in df_vclaim_top_depth.iterrows():
            vclaim_id = elem[cf.DOC_NO]
            vclaim_text = self.df_claim.at[vclaim_id, cf.VCLAIM]
            vclaim_title = self.df_claim.at[vclaim_id, cf.TITLE]
            rank = df_one_query_run.at[vclaim_id, cf.RANK] + 1 # rank is zero-based, so we need to add 1 
            score = df_one_query_run.at[vclaim_id, cf.SCORE]
            if vclaim_id in df_qrels_one[cf.DOC_NO].values: # if this vclaim exists in qrels, then it is a positive example and no need to add it
                continue
            if add_similarity:
                lexical_similarity = df_one_query_run.at[vclaim_id, cf.LEXICAL_SIMILARITY]
                vclaim_semantic_similarity = self.predict_similarity(tweet_text, vclaim_text)
                title_semantic_similarity = self.predict_similarity(tweet_text, vclaim_title)

            vclaim_row = self.get_one_row(tweet_id, tweet_text, vclaim_id, vclaim_text, 
                                        cf.NEGATIVE_LABEL, rank, score, lexical_similarity, vclaim_semantic_similarity)   

            title_row = self.get_one_row(tweet_id, tweet_text, vclaim_id, vclaim_title, 
                                    cf.NEGATIVE_LABEL, rank, score, lexical_similarity, title_semantic_similarity)       

            if what_to_add == cf.TITLE_ONLY:
                # Add the verified claim title only                
                df_result = df_result.append(title_row, ignore_index=True)
            
            else:
                # Add the verified claim text
                df_result = df_result.append(vclaim_row, ignore_index=True)

            k = k + 1
            if k == target: 
                break

        return df_result


    # for a given tweet_id, bring all positive examples from qrels file,
    # and add equal number of negative examples from the run
    def add_positive_and_negative_with_title(self, query_column, df_query, tweet_id, df_qrels_one, df_run, df_result, 
                                add_similarity=False, depth_of_random=20):
        
        df_one_query_run = df_run[df_run[cf.TWEET_ID] == tweet_id]
        df_vclaim_top_depth = df_one_query_run.head(depth_of_random)
        df_vclaim_top_depth = df_vclaim_top_depth.sample(frac=1) # shuffle the dataframe to choose random negative examples

        tweet_text = df_query.at[tweet_id, query_column]
        lexical_similarity = -1
        semantic_similarity = -1
        
        target = len(df_qrels_one)
        df_one_query_run.set_index(cf.DOC_NO, inplace=True)

        # 1- Add positive examples from qrels file.
        # here df_qrels_one contains all rows that corresponds to one tweet only
        # so the tweet_id value will not change in the for loop
        for i, elem in df_qrels_one.iterrows():
            vclaim_id = elem[cf.DOC_NO]
            vclaim_text = self.df_claim.at[vclaim_id, cf.VCLAIM]
            vclaim_title = self.df_claim.at[vclaim_id, cf.TITLE]

            if vclaim_id in df_one_query_run.index: 
                rank = df_one_query_run.at[vclaim_id, cf.RANK] + 1 # rank is zero-based, so we need to add 1 
                score = df_one_query_run.at[vclaim_id, cf.SCORE]
                if add_similarity: #  if want to add lexical and semantic similarities
                    lexical_similarity = df_one_query_run.at[vclaim_id, cf.LEXICAL_SIMILARITY]
                    semantic_similarity = df_one_query_run.at[vclaim_id, cf.SEMANTIC_SIMILARITY]
            else: # relevant document might not be within retrieved documents, so give it the least value
                rank = self.bm25_search_depth
                score = df_one_query_run[cf.SCORE].min()
                if add_similarity:
                    lexical_similarity = cf.EPSILON
                    semantic_similarity = self.predict_similarity(tweet_text, vclaim_text)
            # Add the verified claim text and its title
            new_row = self.get_one_row(tweet_id, tweet_text, vclaim_id, vclaim_text, 
                                        cf.POSITIVE_LABEL, rank, score, lexical_similarity,
                                        semantic_similarity, title=vclaim_title)
            df_result = df_result.append(new_row, ignore_index=True)


        # 2- Add negative examples by same number of positive examples
        # for the current tweet, go through the run and choose the top 10 vclaim
        # which are NOT included as positive examples, choose them as negative examples
        
        k = 0
        for i, elem in df_vclaim_top_depth.iterrows():
            vclaim_id = elem[cf.DOC_NO]
            vclaim_text = self.df_claim.at[vclaim_id, cf.VCLAIM]
            vclaim_title = self.df_claim.at[vclaim_id, cf.TITLE]
            rank = df_one_query_run.at[vclaim_id, cf.RANK] + 1 # rank is zero-based, so we need to add 1 
            score = df_one_query_run.at[vclaim_id, cf.SCORE]
            if vclaim_id in df_qrels_one[cf.DOC_NO].values: # if this vclaim exists in qrels, then it is a positive example and no need to add it
                continue
            if add_similarity:
                lexical_similarity = df_one_query_run.at[vclaim_id, cf.LEXICAL_SIMILARITY]
                semantic_similarity = df_one_query_run.at[vclaim_id, cf.SEMANTIC_SIMILARITY]
            new_row = self.get_one_row(tweet_id, tweet_text, vclaim_id, vclaim_text, 
                                        cf.NEGATIVE_LABEL, rank, score, lexical_similarity, 
                                        semantic_similarity, title=vclaim_title)
            df_result = df_result.append(new_row, ignore_index=True)
            k = k + 1
            if k == target: 
                break

        return df_result



    # this file creates the training dataset
    def create_train_pairs(self, query_path, run_file, data_save_path, query_column="cleaned", what_to_add=cf.VCLAIM_AND_TITLE, 
                            depth_of_random=20, add_similarity=False,):

        df_query = Utils.read_file(query_path)
        df_query[cf.TWEET_ID] = df_query[cf.TWEET_ID].astype(str)
        df_query.set_index(cf.TWEET_ID, inplace=True)

        if add_similarity: # if add similarity is true, then lexical and semantic similarity columns are added to run file
            df_run = Utils.read_file(run_file)
            df_run[cf.DOC_NO] = df_run[cf.VCLAIM_ID].astype(str)
        else:
            df_run = pd.read_csv(run_file, sep="\t", names=[cf.TWEET_ID, cf.DOCID, cf.DOC_NO, cf.RANK, cf.SCORE,cf.QUERY],)
            df_run[cf.DOC_NO] = df_run[cf.DOC_NO].astype(str)

        df_result = pd.DataFrame(columns=[cf.TWEET_ID, cf.TWEET_TEXT, cf.VCLAIM_ID, cf.VCLAIM, cf.LABEL, 
                                cf.RANK, cf.SCORE, cf.LEXICAL_SIMILARITY, cf.SEMANTIC_SIMILARITY])

        df_run[cf.TWEET_ID] = df_run[cf.TWEET_ID].astype(str)

        for i, tweet_id in enumerate(df_run[cf.TWEET_ID].unique()): 

            if i % 100 == 0:
                print("Processing tweet number {} with tweet id {} ".format(i, tweet_id))
            df_qrel_for_one_tweet = self.df_qrels[self.df_qrels[cf.QID] == tweet_id]
            df_result = self.add_positive_and_negative(query_column, df_query, tweet_id, df_qrel_for_one_tweet, df_run, df_result, 
                            what_to_add, depth_of_random=depth_of_random, add_similarity=add_similarity,)
        
        if not os.path.isfile(data_save_path): # create the directory if it does not exist
            os.makedirs(os.path.dirname(data_save_path), exist_ok=True)

        df_result.to_excel(data_save_path, index=False, encoding="utf-8")
        return df_result
    
    #Passing a query is optional
    def create_test_pairs_and_rerank(self, query_path, query_column, qrels_path, hp, depth = 20, bm25_run_path="", 
                                retrieval_model="BM25", mono_bert_pairs_path="", reranked_pairs_path="", 
                                evaluation_save_path="", trec_run_path="", lang="en", what_to_test=cf.VCLAIM_ONLY,):
        try:  
            
            print("Run retrieval and evaluation for depth ", depth)
            self.search_and_evaluate(query_path=query_path, evaluation_path="", retrieval_model=retrieval_model,
                                    run_save_path=bm25_run_path, depth=depth, data_column=query_column,
                                    lang=lang)

            # 2. Create test set
            # if not os.path.isfile(mono_bert_pairs_path):
            print("Creating test pairs in mono BERT fashion for the queries in path: ", query_path)
            self.create_test_pairs(query_path, bm25_run_path, query_column, pairs_save_path=mono_bert_pairs_path)
            print("Done creating test pairs")

            print("Predicting relevance scores for pairs and rerank them accordingly ...")
            mono_bert_tester = MonoBertTester(qrels_path=qrels_path,
                                evaluation_save_path=evaluation_save_path)
            mono_bert_tester.test_mono_bert(hp["model_name"], hp["model_save_path"], mono_bert_pairs_path, 
                                reranked_pairs_path,  evaluation_save_path, max_len=hp["max_len"], batch_size=hp["batch_size"],
                                dropout=hp["dropout"], is_output_probability=hp["is_output_probability"], 
                                hyper_parameters=hp,  classifier_layers=hp["num_of_layers"],
                                trec_run_path=trec_run_path, what_to_test=what_to_test, qrels_path=qrels_path)


        except Exception as e:
            logging.error('error occured at retrieve_relevant_vclaims: {}'.format(e))
            return []

    

    def merge_lexical_and_semantic_scores(self, mono_bert_train_pairs_path, reranked_pairs_path, merged_pairs_path):
        '''
        Normalize BM25 scores (lexical similarity) for each query-document pair and merge them with
        their corresponding semantic score (reranking score produced by mono bert reranker)
        '''     
        reranking_scores = []
        bm25_normalized_scores = []

        df_train_pairs = Utils.read_file(mono_bert_train_pairs_path)
        df_reranked_pairs = Utils.read_file(reranked_pairs_path)
        df_reranked_pairs.set_index([cf.TWEET_ID, cf.VCLAIM_ID], inplace=True)
        print("df_train_pairs.columns", df_train_pairs.columns)
        print("df_reranked_pairs.columns", df_reranked_pairs.columns)

        df_new_train_pairs = df_train_pairs.copy() # take copy of the original training pairs
        df_train_pairs[cf.LEXICAL_SIMILARITY] = [0.0] * len(df_train_pairs) # add new columns
        df_train_pairs[cf.SEMANTIC_SIMILARITY] = [0.0] * len(df_train_pairs)
        df_new_train_pairs.set_index([cf.TWEET_ID, cf.VCLAIM_ID], inplace=True)

        for tweet_id in df_train_pairs[cf.TWEET_ID].unique(): # iterate over unique tweet ids
            df_one_tweet_pairs = df_train_pairs[df_train_pairs[cf.TWEET_ID] == tweet_id] # get rows that has this tweet-id
            max_bm25_score = df_one_tweet_pairs[cf.SCORE].max() 
            min_bm25_score = df_one_tweet_pairs[cf.SCORE].min()
            max_min_range = max_bm25_score - min_bm25_score # to save computation cost 

            for i, row in df_one_tweet_pairs.iterrows():
                vclaim_id = row[cf.VCLAIM_ID]
                bm25_score = row[cf.SCORE]
                bm25_normalized_score  = (bm25_score - min_bm25_score) / max_min_range # compute the normalized bm25 score
                bm25_normalized_scores.append(bm25_normalized_score)

                reranking_score = df_reranked_pairs.loc[tweet_id, vclaim_id][cf.SCORE] # get the corresponding reranking score produced by the reranker
                reranking_scores.append(reranking_score)

                df_new_train_pairs.at[(tweet_id, vclaim_id), cf.LEXICAL_SIMILARITY] = bm25_normalized_score
                df_new_train_pairs.at[(tweet_id, vclaim_id), cf.SEMANTIC_SIMILARITY] = reranking_score
                # print("bm25_normalized_score is ", bm25_normalized_score)
                # print("reranking_score is ", reranking_score)

        
        df_new_train_pairs.reset_index(inplace=True,)
        df_new_train_pairs.to_excel(merged_pairs_path, index=False,)
        return df_new_train_pairs

    
    def load_semantic_model(self, model_name, trained_model_weights, classifier_layers, num_classes, 
                             dropout, is_output_probability, freeze_bert=False, max_len=256):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # load the fined-tuned model
        if classifier_layers == cf.TWO_LAYERS:
            model = classifier.RelevanceClassifierTwoLayers(bert_name=model_name, n_classes=num_classes, 
                            freeze_bert=freeze_bert, dropout=dropout,is_output_probability=is_output_probability)
        else:
            model = classifier.RelevanceClassifierOneLayer(bert_name=model_name, n_classes=num_classes,
                            freeze_bert=freeze_bert, dropout=dropout,is_output_probability=is_output_probability)

        model.load_state_dict(torch.load(trained_model_weights, map_location=torch.device(self.device)))
        model = model.to(self.device)

        self.is_output_probability = is_output_probability
        self.max_len= max_len
        self.sementic_model = model
        self.tokenizer = tokenizer
        return 
    
    def predict_similarity(self, tweet, vclaim, ):
        indices = torch.tensor([1]).to(self.device)
        encoding = self.tokenizer.encode_plus(
            tweet,
            vclaim,
            add_special_tokens=True,
            max_length= self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt", )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        if self.is_output_probability:
            probs = self.sementic_model(input_ids=input_ids, attention_mask=attention_mask) # outputs are probabilities of each class
        else:
            logits = self.sementic_model(input_ids=input_ids, attention_mask=attention_mask) # outputs are logits
            probs = F.softmax(logits, dim=1) # needed if the output are logits

        _, preds = torch.max(probs, dim=1)
        # choose the neuron that predics the relevance score
        probs = torch.index_select(probs, dim=1, index=indices)

        return probs.item()

    # Create train (query-vclaim) pairs and title and other fields to the training pairs
    def create_train_pairs_with_title(self, query_path, query_column, data_save_path, bm25_run_path="", 
                            retrieval_model="BM25", add_similarity=False, search_depth=100, depth_of_random=20,
                            lang="en"):
        try:  

            print("Run retrieval and evaluation for depth ", search_depth)
            self.search_and_evaluate(query_path=query_path, evaluation_path="", retrieval_model=retrieval_model,
                                    run_save_path=bm25_run_path, depth=search_depth, data_column=query_column,
                                    lang=lang)

            # 2. Create test set
            print("Creating test pairs in mono BERT fashion for the queries in path: ", query_path)

            df_query = Utils.read_file(query_path)
            df_query[cf.TWEET_ID] = df_query[cf.TWEET_ID].astype(str)
            df_query.set_index(cf.TWEET_ID, inplace=True)

            if add_similarity: # if add similarity is true, then lexical and semantic similarity columns are added to run file
                df_run = Utils.read_file(bm25_run_path)
                df_run[cf.DOC_NO] = df_run[cf.VCLAIM_ID].astype(str)
            else:
                df_run = pd.read_csv(bm25_run_path, sep="\t", names=[cf.TWEET_ID, cf.DOCID, cf.DOC_NO, cf.RANK, cf.SCORE,cf.QUERY],)
                df_run[cf.DOC_NO] = df_run[cf.DOC_NO].astype(str)

            df_result = pd.DataFrame(columns=[cf.TWEET_ID, cf.TWEET_TEXT, cf.VCLAIM_ID, cf.VCLAIM, cf.TITLE, cf.LABEL, 
                                    cf.RANK, cf.SCORE, cf.LEXICAL_SIMILARITY, cf.SEMANTIC_SIMILARITY])

            df_run[cf.TWEET_ID] = df_run[cf.TWEET_ID].astype(str)

            for i, tweet_id in enumerate(df_run[cf.TWEET_ID].unique()): 

                if i % 100 == 0:
                    print("Processing tweet number {} with tweet id {} ".format(i, tweet_id))
                df_for_one_tweet = self.df_qrels[self.df_qrels[cf.QID] == tweet_id]
                df_result = self.add_positive_and_negative_with_title(query_column, df_query, tweet_id, df_for_one_tweet, 
                                                                    df_run, df_result, depth_of_random=depth_of_random, 
                                                                    add_similarity=add_similarity,)
                

            df_result.to_excel(data_save_path, index=False, encoding="utf-8")
            return df_result


        except Exception as e:
            logging.error('error occured while creating train pairs with title: {}'.format(e))
            return []
            
    
    def convert_mono_run_to_pt_run(self, run_path, save_path):
        df = Utils.read_file(run_path)
        df[cf.DOCID] = df[cf.VCLAIM_ID]
        df[cf.DOC_NO] = df[cf.VCLAIM_ID]
        df[cf.QUERY] = df[cf.TWEET_TEXT] 

        new_df = pd.DataFrame()
        for tweet_id in df[cf.TWEET_ID].unique():
            df_one_query = df[df[cf.TWEET_ID] == tweet_id].copy()
            df_one_query[cf.RANK] = list(range(len(df_one_query)))
            new_df = new_df.append(df_one_query, ignore_index=False)
        new_df = new_df[[cf.TWEET_ID, cf.DOCID, cf.DOC_NO, cf.RANK, cf.SCORE, cf.QUERY]]
        new_df.to_csv(save_path,  header=False, index=False, sep='\t', encoding="utf-8")
        return new_df 


