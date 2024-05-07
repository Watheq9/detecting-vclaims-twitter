# import needed libraries
import pandas as pd
import torch
import helper.Utils as Utils
import helper.dataset as dataset
import helper.classifier as classifier
import datetime
import time
import csv
import numpy as np
import math
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch import nn
import csv
import torch
import pandas as pd 
from pyterrier.measures import RR, R, Rprec, P, MAP
import pyterrier as pt
from statistics import harmonic_mean
import configure as cf
from trectools import TrecRun, TrecQrel, TrecEval
import os


class MonoBertBase():

    # define some constants.
    DECIMAL_ROUND = 5
    pos_label = 1.0
    neg_label = 0.0
    RANK = cf.RANK
    SCORE = cf.SCORE
    TWEET_ID_COLUMN = cf.TWEET_ID
    TWEET_TEXT_COLUMN = cf.TWEET_TEXT
    VCLAIM_ID = cf.VCLAIM_ID
    VCLAIM = cf.VCLAIM
    TITLE = cf.TITLE 
    LABEL = cf.LABEL
    TAG = cf.TAG

    QUERY = cf.QUERY
    QID = cf.QID
    DOC_NO = cf.DOC_NO
    DOCID = cf.DOCID
    NUM_CLASSES = 2
    Q0 = "Q0"
    NORMAL_CURRICULA = cf.NORMAL_CURRICULA
    ALL_RELEVANT_CURRICULA = cf.ALL_RELEVANT_CURRICULA
    ALL_NON_RELEVANT_CURRICULA = cf.ALL_NON_RELEVANT_CURRICULA
    HARMONIC_MEAN_OF_SIMILARITY = cf.HARMONIC_MEAN_OF_SIMILARITY
    HARMONIC_MEAN_OF_RECIP_RANK_AND_SIMILARITY = cf.HARMONIC_MEAN_OF_RECIP_RANK_AND_SIMILARITY
    

    def __init__(self, qrels_file, eval_metrics):
        self.gold_labels = TrecQrel(qrels_file)
        self.eval_metrics=eval_metrics
        if not pt.started():
            print("Enabling PRF in pyterier")
            # In this lab, we need to specify that we start PyTerrier with PRF enabled
            pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
    

    def format_time(self, elapsed):
        """
        Takes a time in seconds and returns a string hh:mm:ss
        """
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss

    
    def evaluate_reranking(self, input_var, trec_run_path, qrels_path=""):
        if isinstance(input_var, pd.DataFrame):
            df = input_var
        else: # it is path for a run 
            df = Utils.read_file(input_var)

        # evaluate using pyterier
        df[self.QID] = df[self.TWEET_ID_COLUMN].astype(str)
        df[self.DOC_NO] = df[self.VCLAIM_ID].astype(str)
        prediction = TrecRun(trec_run_path)

        if qrels_path != "":
            df_qrels = pd.read_csv(qrels_path, sep="\t", names=[self.QID, self.Q0, self.DOC_NO, self.LABEL])
            gold_labels = TrecQrel(qrels_path)
            eval_res = pt.Utils.evaluate(df, df_qrels[[self.QID, self.DOC_NO, self.LABEL]],metrics=self.eval_metrics)
            trec_eval_metrics = TrecEval(prediction, gold_labels)

        else:
            eval_res = pt.Utils.evaluate(df, self.df_qrels[[self.QID, self.DOC_NO, self.LABEL]],metrics=self.eval_metrics)
            trec_eval_metrics = TrecEval(prediction, self.gold_labels)
        
        # evaluate using trec tools (trec eval)
        eval_res.update({"map": trec_eval_metrics.get_map(), 
                        "AP@5": trec_eval_metrics.get_map(depth=5),
                        "P@1": trec_eval_metrics.get_precision(depth=1),
                        "RR": trec_eval_metrics.get_reciprocal_rank(),
                        "Rprec": trec_eval_metrics.get_rprec()})

        return eval_res


    def re_rank_and_save_output(self, tweet_ids, tweets, vclaims, vclaim_ids, probs, labels, 
            save_path="", trec_run_path=""):

        '''
        save_path: path to save the reranked results with queries and documents text
        trec_run_path: path for saving the reranked results int trec eval format 
        '''

        t0 = time.time()
        df = pd.DataFrame()

        df[cf.TWEET_ID] = tweet_ids
        df[cf.TWEET_TEXT] = tweets
        df[cf.VCLAIM_ID] = vclaim_ids
        df[cf.VCLAIM] = vclaims
        df[cf.SCORE] = probs
        df[cf.LABEL] = labels

        df[cf.TWEET_ID] = df[cf.TWEET_ID].astype(str)
        df[cf.VCLAIM_ID] = df[cf.VCLAIM_ID].astype(str)
        # 1. group by tweet id, then sort based on the score value
        df = df.groupby([cf.TWEET_ID]).apply(lambda x: x.sort_values([cf.SCORE], 
                                    ascending = False)).reset_index(drop=True)
        print("Done with re-ranking tweets ")


        df_trec = pd.DataFrame(columns = [cf.TWEET_ID, cf.Q0, cf.VCLAIM_ID, cf.RANK, cf.SCORE, cf.TAG])
        df_new = pd.DataFrame(columns = [cf.TWEET_ID, cf.TWEET_TEXT, cf.VCLAIM_ID, cf.VCLAIM,
                                        cf.RANK, cf.SCORE, cf.LABEL])
        for qid in df[cf.TWEET_ID].unique():
            df_one = df[df[cf.TWEET_ID] == qid]
            df_one = df_one.sort_values([cf.SCORE], ascending=False)
            df_one[cf.RANK] = [(i) for i in range(len((df_one)))]
            df_new = df_new.append(df_one[[cf.TWEET_ID, cf.TWEET_TEXT, cf.VCLAIM_ID, cf.VCLAIM,
                                        cf.RANK, cf.SCORE, cf.LABEL]], ignore_index=True)
            
            df_one[cf.RANK] = [(i+1) for i in range(len((df_one)))]
            df_one[cf.Q0] = [cf.Q0] * len(df_one)
            df_one[cf.TAG] = ["reranked_run"] * len(df_one)
            df_trec = df_trec.append(df_one[[cf.TWEET_ID, cf.Q0, cf.VCLAIM_ID, cf.RANK, 
                                        cf.SCORE, cf.TAG]], ignore_index=True)

        if save_path != "":
            if not os.path.isfile(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df_new.to_csv(save_path, index=False, sep='\t', encoding="utf-8")
        
        if not os.path.isfile(trec_run_path): # create the directory if it does not exist
            os.makedirs(os.path.dirname(trec_run_path), exist_ok=True)
        df_trec.to_csv(trec_run_path, sep="\t", index=False, header=False, encoding="utf-8")
        print("Output is saved into ", save_path)
        print("Trec run is saved into ", trec_run_path)

        return df
    


class MonoBertTrainer(MonoBertBase):


    def __init__(self, train_query_path="", dev_query_path="", qrels_path="", 
                    eval_metrics=["map",MAP@5, P@1, RR, Rprec, R@5, R@10, R@20, R@50, RR@5]):
        super().__init__(qrels_path, eval_metrics)
        # self.train_query_path = train_query_path
        # self.dev_query_path = dev_query_path
        self.df_qrels = pd.read_csv(qrels_path, sep="\t", names=[self.QID, self.Q0, self.DOC_NO, self.LABEL])
        self.df_qrels[self.QID]=self.df_qrels[self.QID].astype(str)
        self.df_qrels[self.DOC_NO]= self.df_qrels[self.DOC_NO].astype(str)
    
    

    def get_difficulty(self, labels, ranks, curricula_type, device, lexical_similarity=None, semantic_similarity=None):
        is_relevant = (labels > 0)
        recip_ranks_one = torch.Tensor([1] * len(ranks)).to(device)
        recip_ranks_zero = torch.Tensor([0] * len(ranks)).to(device)
        recip_rank = (1. / ranks)
        if  curricula_type == self.ALL_RELEVANT_CURRICULA:
            difficulty = torch.where(is_relevant, recip_ranks_one, recip_ranks_zero)

        elif curricula_type == self.ALL_NON_RELEVANT_CURRICULA:
            difficulty = torch.where(is_relevant, recip_ranks_zero, recip_ranks_one)
        
        elif curricula_type == self.HARMONIC_MEAN_OF_SIMILARITY:
            # compute the harmonic mean between each similarity  pair and add epislon to avoid zero issues
            my_harmonic_mean = [harmonic_mean([lexical_similarity[i].item()+cf.EPSILON, 
                                            semantic_similarity[i].item()+cf.EPSILON]) for i in range(len(lexical_similarity))]
            harmonic_mean_tensor = torch.Tensor(my_harmonic_mean).to(device)
            difficulty = torch.where(is_relevant, harmonic_mean_tensor, 1. - harmonic_mean_tensor)

        elif curricula_type == self.HARMONIC_MEAN_OF_RECIP_RANK_AND_SIMILARITY:
            # compute the harmonic mean between the reciprocal rank and semantic similarity 
            # and add epislon to avoid zero issues
            my_harmonic_mean = [harmonic_mean([recip_rank[i].item()+cf.EPSILON, 
                                            semantic_similarity[i].item()+cf.EPSILON]) for i in range(len(lexical_similarity))]
            harmonic_mean_tensor = torch.Tensor(my_harmonic_mean).to(device)
            difficulty = torch.where(is_relevant, harmonic_mean_tensor, 1. - harmonic_mean_tensor)

        else: # normal curricula
            difficulty = torch.where(is_relevant, recip_rank, 1. - recip_rank)

        difficulty = difficulty + 1e-8 # for smoothing
        return difficulty




    def get_loss_weight(self, difficulty, epoch_number, end_of_curriculum,):
        progress = epoch_number/end_of_curriculum
        loss_weight = difficulty + progress * (1. - difficulty)
        return loss_weight


    def train_epoch(
        self, epoch_number, model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, 
        is_output_probability=True, curricula_type=0, end_of_curriculum=0):

    
        t0 = time.time()
        model = model.train()
        losses = []
        correct_predictions = 0

        y_test = np.array([], dtype=int)
        y_pred = np.array([], dtype=int)

        for step, batch in enumerate(data_loader):  # Progress update every 40 batches.

            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = self.format_time(time.time() - t0)
                # Report progress.
                print("  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step, len(data_loader), elapsed))  

                
            # Unpack this training batch from dataloader.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch[self.LABEL].to(device)

            if is_output_probability:
                probs = model(input_ids=input_ids, attention_mask=attention_mask) # outputs are probabilities of each class
                loss = loss_fn(probs, labels)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask) # outputs are logits
                loss = loss_fn(logits, labels)
                probs = F.softmax(logits, dim=1) # needed if the output are logits

            if epoch_number < end_of_curriculum and curricula_type != 0:
                ranks = batch[self.RANK].to(device)
                if curricula_type == self.HARMONIC_MEAN_OF_SIMILARITY or curricula_type == self.HARMONIC_MEAN_OF_RECIP_RANK_AND_SIMILARITY:
                    lexical_similarity = batch[cf.LEXICAL_SIMILARITY]
                    semantic_similarity = batch[cf.SEMANTIC_SIMILARITY]
                    difficulty = self.get_difficulty(labels, ranks, curricula_type, device, lexical_similarity, semantic_similarity)
                else:
                    difficulty = self.get_difficulty(labels, ranks, curricula_type, device)
                loss_weight = self.get_loss_weight(difficulty, epoch_number, end_of_curriculum)
                loss = loss * loss_weight.mean()


            _, preds = torch.max(probs, dim=1)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            y_test = np.append(y_test, labels.cpu().numpy())
            y_pred = np.append(y_pred, preds.cpu().numpy())

        elapsed = self.format_time(time.time() - t0)
        print("  Total time for one epoch: {:}.".format(elapsed))  
        print("")
        print("  correct_predictions: {0:.2f}".format(correct_predictions.double()))
        print("  Accuracy : {0:.2f}".format(correct_predictions.double() / n_examples))
        print("  Average training loss: {0:.2f}".format(np.mean(losses)))
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)


        # Compute the evalutation metrics
        relevance_pr = precision_score(y_test, y_pred, pos_label=1, average="binary")
        relevance_recall = recall_score(y_test, y_pred, pos_label=1, average="binary")
        relevance_f1 = f1_score(y_test, y_pred, pos_label=1, average="binary")
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        accuracy = accuracy_score(y_test, y_pred)

        print(f"relevance results F1 score  {relevance_f1}  precision {relevance_pr} recall {relevance_recall}")
        print(f" Macro F1 {macro_f1} Weighted F1 {weighted_f1} Accuracy {accuracy}")

        return accuracy, np.mean(losses)


  

        
    def eval_model_binary(self, model, epoch_number, data_loader, loss_fn, device, n_examples, is_output_probability=True,
                    curricula_type=0, end_of_curriculum=0): 
        '''
        if the dev set consists only of two documents for each query, then you can use this function 
        '''
        print("Running Evaluation...")
        t0 = time.time()  # Put the model in evaluation mode--the dropout layers behave differently
        model = model.eval()

        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch[self.LABEL].to(device)

                if is_output_probability:
                    probs = model(input_ids=input_ids, attention_mask=attention_mask) # outputs are probabilities of each class
                    loss = loss_fn(probs, labels)
                else:
                    logits = model(input_ids=input_ids, attention_mask=attention_mask) # outputs are logits
                    loss = loss_fn(logits, labels)
                    probs = F.softmax(logits, dim=1) # needed if the output are logits

                if epoch_number < end_of_curriculum and curricula_type != 0:
                    ranks = batch[self.RANK].to(device)
                    difficulty = self.get_difficulty(labels, ranks, curricula_type, device)
                    loss_weight = self.get_loss_weight(difficulty, epoch_number, end_of_curriculum)
                    loss = loss * loss_weight.mean()

                _, preds = torch.max(probs, dim=1)
                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())

        accuracy = correct_predictions.double() / n_examples
        dev_loss = np.mean(losses)

        print(" \n Done evaluation  -------------------")
        print("  Accuracy: {0:.2f}".format(accuracy))
        print("  Average Validation loss: {0:.2f} \n".format(dev_loss))
        print("  correct_predictions: {0:.2f}".format(correct_predictions.double()))
        print("  n_examples: {0:.2f}".format(n_examples))
        print("  Evaluation took: {:}".format(self.format_time(time.time() - t0)))
        return accuracy, dev_loss, 


    def eval_model(self, model, epoch_number, data_loader, loss_fn, device, n_examples, is_output_probability=True,
                    curricula_type=0, end_of_curriculum=0):
        '''
        if the dev set consists only of top k documents for each query, then you can use this function 
        '''
        print("Running Evaluation...")
        predictions = []
        prediction_probs = []
        all_labels = []
        queries = []
        query_ids = []
        document_ids = []
        documents = []
        losses = []
        indices = torch.tensor([1]).to(device)
        correct_predictions = 0
        t0 = time.time()  # Put the model in evaluation mode--the dropout layers behave differently
        model = model.eval()

        with torch.no_grad():
            for step, batch in enumerate(data_loader):  # Progress update every 40 batches.
                if step % 100 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    # Report progress.
                    print("  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step, len(data_loader), elapsed))  

                query = batch[self.TWEET_TEXT_COLUMN]
                query_id = batch[self.TWEET_ID_COLUMN]
                document = batch[self.VCLAIM]
                document_id = batch[self.VCLAIM_ID]
                labels = batch[self.LABEL].to(device)          
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                if is_output_probability:
                    probs = model(input_ids=input_ids, attention_mask=attention_mask) # outputs are probabilities of each class
                    loss = loss_fn(probs, labels)
                else:
                    logits = model(input_ids=input_ids, attention_mask=attention_mask) # outputs are logits
                    loss = loss_fn(logits, labels)
                    probs = F.softmax(logits, dim=1) # needed if the output are logits


                _, preds = torch.max(probs, dim=1)
                # choose the neuron that predics the relevance score
                probs = torch.index_select(probs, dim=1, index=indices)

                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())


                all_labels.extend(labels)
                prediction_probs.extend(probs.flatten().tolist())
                predictions.extend(preds)
                queries.extend(query)
                query_ids.extend(query_id)
                documents.extend(document)
                document_ids.extend(document_id)

        predictions = torch.stack(predictions).cpu()
        all_labels = torch.stack(all_labels).cpu()

        # 2. rerank based on the new scores
        # save_path="" means no saving 
        trec_run_path = "./data/runs/dev_resuable_trec_run.tsv"
        df = self.re_rank_and_save_output(query_ids, queries, documents, document_ids, prediction_probs, 
                            all_labels, save_path="", trec_run_path=trec_run_path) 
                                            
        # 3. evaluate the reranked data
        eval_measures = self.evaluate_reranking(df, trec_run_path=trec_run_path)
        
        accuracy = correct_predictions.double() / n_examples
        dev_loss = np.mean(losses)


        print(" \n Done evaluation  -------------------")
        # print(eval_measures)
        print("  Map measure : {0:.2f}".format(eval_measures["map"]))
        print("  Accuracy: {0:.2f}".format(accuracy))
        print("  Average Validation loss: {0:.2f} \n".format(dev_loss))
        return accuracy, dev_loss, eval_measures


    def mono_loss(self, logits, labels, num_classes=-1):
        if num_classes == -1:
            num_classes = self.NUM_CLASSES
        log_probabilities = torch.nn.functional.log_softmax(logits, dim=1) # you can set dim = -1  
        labels_tensor  = labels.clone().detach()
        one_hot_labels = torch.nn.functional.one_hot(labels_tensor, num_classes=num_classes)
        per_example_loss = -torch.sum(one_hot_labels * log_probabilities, dim=1)
        loss = torch.mean(per_example_loss)
        # probabilities = torch.nn.functional.softmax(logits, dim=1)
        return loss

    def perform_training(self, num_epochs, model, train_data_loader, dev_data_loader, loss_fn, optimizer,
                        device, scheduler, TRAIN_LENGTH, DEV_LENGTH, is_output_probability, curricula_type, end_of_curriculum):


        for epoch in range(num_epochs):

            epoch = epoch + 1
            print(f"Epoch {epoch}/{num_epochs}")
            print("-" * 10)

            train_acc, train_loss = self.train_epoch(epoch, model, train_data_loader, loss_fn, optimizer, 
                            device, scheduler, TRAIN_LENGTH, is_output_probability, curricula_type, end_of_curriculum)
            print(f"Train loss {train_loss} accuracy {train_acc}")

            if dev_data_loader is None:
                dev_acc, dev_loss, eval_measures = None, None, None
            else:
                dev_acc, dev_loss, eval_measures = self.eval_model(model, epoch, dev_data_loader, loss_fn, device, 
                                            DEV_LENGTH, is_output_probability, curricula_type, end_of_curriculum)
                print(f"Dev loss {dev_loss} accuracy {dev_acc} eval_measures {eval_measures}")
            
        return train_acc, train_loss, dev_acc, dev_loss, eval_measures

    def compare(self, dict1, dict2):
        if dict1["map"] > dict2["map"]:
            return 1 # 1 means dict1 > dict2
        elif dict1["map"] == dict2["map"]:
            if dict1["P@1"] > dict2["P@1"]:
                return 1
            elif dict1["P@1"] == dict2["P@1"]:
                if dict1["R@5"] > dict2["R@5"]:
                    return 1
        return 0
        
    def train_mono_bert(self, train_set_path, dev_set_path, model_name, apply_cleaning=False, 
                        trained_model_save_path="", shuffle=True, freeze_bert=False, max_len=180, 
                        batch_size=32, epochs=[3], learning_rates = [3e-5], seeds=[42], 
                        classifier_layers=cf.TWO_LAYERS, is_output_probability=True, curricula_type=0, 
                        end_of_curriculums=[0], dropout=[0.3], hp={}, results_path="", train_with_title=False,
                        what_to_eval=cf.VCLAIM_ONLY, apply_tuning=True):
        '''
        train_with_title: Flag forwards the model to create data loader with triplet (query, claim, claim_title)
        '''

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ", device)
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        if curricula_type == 0: # no curricula
            end_of_curriculums = [0] # to make them consistent with each other
        
        with open(results_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "num of epochs", "Dropout", "learning rate", "m", "Train Accuracy","Train loss", "Dev accuracy",
                "Dev loss", "Dev MAP", "Dev eval measures", "other hyperparameters" ])

        best_dev_loss = 1.0
        best_dev_acc = -1
        best_learning_rate = -1
        best_num_of_epochs = -1
        best_end_of_curriculum= -1
        best_measures = -1
        best_dropout = -1
                
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        RANDOM_SEED = seeds[0]
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        

        df_train = Utils.read_file(train_set_path)
        if apply_tuning: 
        # if we need to tune the model, then report performance on dev set; otherwise, don't use dev set.
            df_dev = Utils.read_file(dev_set_path)
   

        lexical_similarity = df_train[cf.LEXICAL_SIMILARITY] if cf.LEXICAL_SIMILARITY in df_train.columns else None
        semantic_similarity = df_train[cf.SEMANTIC_SIMILARITY] if cf.SEMANTIC_SIMILARITY in df_train.columns else None

        # create train data loader
        train_data_loader = dataset.create_data_loader(
                            df_train[self.TWEET_TEXT_COLUMN], df_train[self.TWEET_ID_COLUMN], df_train[self.VCLAIM], 
                            df_train[self.VCLAIM_ID], df_train[self.LABEL], tokenizer, max_len, batch_size, 
                            ranks=df_train[self.RANK], lexical_similarity=lexical_similarity, 
                            semantic_similarity=semantic_similarity)

        if apply_tuning is False:
            dev_data_loader = None # no need for evaluation

        # create evaluation data loader
        elif what_to_eval == cf.VCLAIM_AND_TITLE:
            dev_data_loader = dataset.create_data_loader_with_title(
                            df_dev[self.TWEET_TEXT_COLUMN], df_dev[self.TWEET_ID_COLUMN], df_dev[self.VCLAIM], df_dev[self.VCLAIM_ID], 
                            df_dev[self.LABEL], tokenizer, max_len, batch_size, ranks=df_dev[self.RANK], titles=df_dev[self.TITLE])
        
        else:
            if what_to_eval == cf.TITLE_ONLY:
                df_dev_doc = df_dev[self.TITLE]
            else: # vclaim oly
                df_dev_doc = df_dev[self.VCLAIM]
            
            dev_data_loader = dataset.create_data_loader(
                            df_dev[self.TWEET_TEXT_COLUMN], df_dev[self.TWEET_ID_COLUMN], df_dev_doc, df_dev[self.VCLAIM_ID], 
                            df_dev[self.LABEL], tokenizer, max_len, batch_size, ranks=df_dev[self.RANK])

        
        if apply_cleaning:
            df_train[self.TWEET_TEXT_COLUMN] = df_train[self.TWEET_TEXT_COLUMN].apply(Utils.clean)
            if apply_tuning:
                df_dev[self.TWEET_TEXT_COLUMN] = df_dev[self.TWEET_TEXT_COLUMN].apply(Utils.clean)
        
        # shuffle the data
        if shuffle:
            df_train = df_train.sample(frac=1, random_state=seeds[0])
            if apply_tuning:
                df_dev = df_dev.sample(frac=1, random_state=seeds[0])
        
        for epoch in epochs: # could be [2,3,4] or others
            for learning_rate in learning_rates: # values should be in range [2e-5, 5e-5]
                for dropout_value in dropout:
                    for end_of_curriculum in end_of_curriculums:
                    
                        if end_of_curriculum > epoch:
                            break

                        TRAIN_LENGTH = len(df_train[self.TWEET_TEXT_COLUMN])
                        if apply_tuning:
                            DEV_LENGTH = len(df_dev[self.TWEET_TEXT_COLUMN])
                        else:
                            DEV_LENGTH = 0

                        print("train size ", TRAIN_LENGTH)
                        print("dev size ", DEV_LENGTH)

                        # -----------------  Initialize the classifier  ----------
                        if classifier_layers == cf.TWO_LAYERS:
                            model = classifier.RelevanceClassifierTwoLayers(bert_name=model_name, n_classes=self.NUM_CLASSES,
                                    freeze_bert=freeze_bert, dropout=dropout_value,is_output_probability=is_output_probability)
                        else:
                            model = classifier.RelevanceClassifierOneLayer(bert_name=model_name, n_classes=self.NUM_CLASSES, 
                                    freeze_bert=freeze_bert, dropout=dropout_value,is_output_probability=is_output_probability)
                        model = model.to(device)

                        optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
                        total_steps = len(train_data_loader) * epoch
                        warmup_steps = math.ceil(len(train_data_loader) * epoch * 0.1)
                        
                        if is_output_probability:
                            loss_fn = nn.CrossEntropyLoss().to(device)
                        else:
                            loss_fn = self.mono_loss

                        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_steps)

                        train_acc, train_loss, dev_acc, dev_loss, eval_measures = self.perform_training(epoch, model, 
                                                        train_data_loader, dev_data_loader, loss_fn, optimizer,
                                                        device, scheduler, TRAIN_LENGTH, DEV_LENGTH, 
                                                        is_output_probability, curricula_type, end_of_curriculum)
                        
                        if apply_tuning is False: # if no tuning, store the last model
                            best_learning_rate = learning_rate
                            best_num_of_epochs = epoch
                            best_end_of_curriculum = end_of_curriculum
                            best_dropout = dropout_value
                            torch.save(model.state_dict(), trained_model_save_path)
                            
                        elif best_measures == -1 or self.compare(eval_measures, best_measures) > 0:
                            best_measures = eval_measures
                            best_dev_loss = dev_loss
                            best_dev_acc = dev_acc
                            best_learning_rate = learning_rate
                            best_num_of_epochs = epoch
                            best_end_of_curriculum = end_of_curriculum
                            best_dropout = dropout_value
                            torch.save(model.state_dict(), trained_model_save_path)
                        
                        

                        # printing current measures and save them into results file
                        print(f" -------------------Training complete for learning rate {learning_rate} and number of epochs {epoch} ------ ")
                        print(f"Train loss {train_loss} accuracy {train_acc} end_of_curriculum {end_of_curriculum}")
                        print(f"Dev loss {dev_loss} dev {dev_acc} ")
                        print(f"\n ------------------------------------------------------------ ")
                        with open(results_path, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([epoch, dropout_value, learning_rate, end_of_curriculum, train_acc,train_loss, dev_acc,
                                            dev_loss, eval_measures, hp])


        print(f" ------------------- Overall Training complete! ---------------------------------------")
        print(f"Best learning rate {best_learning_rate} and number of epochs {best_num_of_epochs} end_of_curriculum {best_end_of_curriculum} ------ ")
        print(f"Best Dev loss {best_dev_loss} and dev accuracy  {best_dev_acc} best map {best_measures}")
        print(f"\n ------------------------------------------------------------ ")
        
        with open(results_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([best_num_of_epochs, best_dropout, best_learning_rate, 
                            best_end_of_curriculum, "best-hp","best-hp", best_dev_acc,
                            best_dev_loss, best_measures, hp])

        return best_dev_loss, best_dev_acc, best_learning_rate, best_num_of_epochs, best_end_of_curriculum, best_dropout

