# import needed libraries
import pandas as pd
import torch
import helper.Utils as Utils
import helper.dataset as dataset
import helper.classifier as classifier
import time
import torch.nn.functional as F
from transformers import AutoTokenizer
import torch
import pandas as pd
import os
from mono_bert_train import MonoBertBase
import configure as cf
from pyterrier.measures import RR, R, Rprec, P, MAP
import pyterrier as pt

class MonoBertTester(MonoBertBase):


    def __init__(self, qrels_path, evaluation_save_path, 
                eval_metrics=["map",MAP@5, P@1, RR, Rprec, R@5, R@10, R@20, R@50, RR@5]):
        super().__init__(qrels_path, eval_metrics)
        if qrels_path != "":
            self.df_qrels = pd.read_csv(qrels_path, sep="\t", names=[self.QID, self.Q0, self.DOC_NO, self.LABEL])
            self.df_qrels[self.QID]=self.df_qrels[self.QID].astype(str)
            self.df_qrels[self.DOC_NO]= self.df_qrels[self.DOC_NO].astype(str)
        self.evaluation_save_path = evaluation_save_path


    # save the results with hyperparameters as new row within a excel sheet
    def save_results(self, eval_results, hyper_parameters, save_path):
        if save_path =="":
            return 
            
        new_eval_row = dict(list(eval_results.items()) + list(hyper_parameters.items())) # merging two dict
        
        if not os.path.isfile(save_path): # if file is not exist, create a new one
            os.makedirs(os.path.dirname(save_path), exist_ok=True) # create the directory if it does not exist
            df_result = pd.DataFrame([new_eval_row])
        else:
            df_result = Utils.read_file(save_path)
            df_result = df_result.append(new_eval_row, ignore_index=True)
        
        df_result = df_result.round(self.DECIMAL_ROUND)
        df_result.to_excel(save_path, index=False)
        return df_result


    # get predictions of the test set pairs 
    def get_predictions(self, model, data_loader, device, is_output_probability=True, add_title_prediction=False):

        print("Running Testing ...")
        predictions = []
        prediction_probs = []
        labels = []
        queries = []
        query_ids = []
        document_ids = []
        documents = []
        indices = torch.tensor([1]).to(device)
        t0 = time.time()

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
                label = batch[self.LABEL].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                if is_output_probability:
                    probs = model(input_ids=input_ids, attention_mask=attention_mask) # outputs are probabilities of each class
                else:
                    logits = model(input_ids=input_ids, attention_mask=attention_mask) # outputs are logits
                    probs = F.softmax(logits, dim=1) # needed if the output are logits

                _, preds = torch.max(probs, dim=1)
                # choose the neuron that predics the relevance score
                probs = torch.index_select(probs, dim=1, index=indices)

                # Add the score of the query-title pair
                if add_title_prediction:
                    input_ids = batch[cf.QUERY_AND_TITLE_INPUT_IDS].to(device)
                    attention_mask = batch[cf.QUERY_AND_TITLE_ATTENTION_MASK].to(device)
                    if is_output_probability:
                        title_probs = model(input_ids=input_ids, attention_mask=attention_mask) # outputs are probabilities of each class
                    else:
                        logits = model(input_ids=input_ids, attention_mask=attention_mask) # outputs are logits
                        title_probs = F.softmax(logits, dim=1) # needed if the output are logits

                    # choose the neuron that predics the relevance score
                    title_probs = torch.index_select(title_probs, dim=1, index=indices)
                    # merge the probs of query-document and query-title into one tensor
                    probs = torch.cat((title_probs, probs), dim=1)
                    # then take the average
                    probs = torch.mean(probs, dim=1, keepdim=True)
                

                labels.extend(label)
                prediction_probs.extend(probs.flatten().tolist())
                predictions.extend(preds)
                queries.extend(query)
                query_ids.extend(query_id)
                documents.extend(document)
                document_ids.extend(document_id)

        elapsed = self.format_time(time.time() - t0)
        print("  Total time for testing: {:}.".format(elapsed)) 

        predictions = torch.stack(predictions).cpu()
        labels = torch.stack(labels).cpu()
        return queries, query_ids, documents, document_ids, predictions, prediction_probs, labels


                            
    def test_mono_bert(self, model_name, trained_model_weights, test_set_path, reranked_run_save_path, evaluation_save_path, 
                        apply_cleaning=False, max_len=180, batch_size=32, is_output_probability=True, 
                        hyper_parameters={}, classifier_layers=cf.TWO_LAYERS, dropout=0.3,freeze_bert=False,
                        add_title_prediction=False, what_to_test=cf.VCLAIM_ONLY,
                        trec_run_path="", qrels_path=""):
        '''
        Run the trained model on the test set, then re-ranked the examples based on the new computed score

        add_title_to_dataloader: Flag forwards the model to create data loader of triplets (query, vclaim, vclaim_title)
        
        '''
        #add_title_prediction is a flag indicates whether the model should add the prediction score of 
        # the query-title pair to the query-claim pair
        add_title_prediction = False 
        eval_measures = []


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # load the fined-tuned model
        if classifier_layers == cf.TWO_LAYERS:
            model = classifier.RelevanceClassifierTwoLayers(bert_name=model_name, n_classes=self.NUM_CLASSES, 
                            freeze_bert=freeze_bert, dropout=dropout,is_output_probability=is_output_probability)
        else:
            model = classifier.RelevanceClassifierOneLayer(bert_name=model_name, n_classes=self.NUM_CLASSES, 
                            freeze_bert=freeze_bert, dropout=dropout,is_output_probability=is_output_probability)

        model.load_state_dict(torch.load(trained_model_weights, map_location=torch.device(device)))
        model = model.to(device)

        # initiate test dataloader 
        df_test = Utils.read_file(test_set_path)
        if apply_cleaning:
            df_test[self.TWEET_TEXT_COLUMN] = df_test[self.TWEET_TEXT_COLUMN].apply(Utils.clean)
        

        # create evaluation data loader
        if what_to_test == cf.VCLAIM_AND_TITLE:
            add_title_prediction = True
            test_data_loader = dataset.create_data_loader_with_title(
                        df_test[self.TWEET_TEXT_COLUMN], df_test[self.TWEET_ID_COLUMN], df_test[self.VCLAIM], 
                        df_test[self.VCLAIM_ID], df_test[self.LABEL], tokenizer, max_len, batch_size, 
                        titles= df_test[cf.TITLE])
        else:
            if what_to_test == cf.TITLE_ONLY:
                df_doc = df_test[self.TITLE]
            else:
                df_doc = df_test[self.VCLAIM]

            test_data_loader = dataset.create_data_loader(
                        df_test[self.TWEET_TEXT_COLUMN], df_test[self.TWEET_ID_COLUMN], df_doc, 
                        df_test[self.VCLAIM_ID], df_test[self.LABEL], tokenizer, max_len, batch_size,)


        # 1. get predictions probabilities for test set
        queries, query_ids, documents, document_ids, predictions, prediction_probs, labels = self.get_predictions(
                model, test_data_loader, device, is_output_probability, add_title_prediction)

        # 2. rerank based on the new scores
        self.re_rank_and_save_output(query_ids, queries, documents, document_ids, prediction_probs, labels, 
                                    save_path=reranked_run_save_path, trec_run_path=trec_run_path) 
                                            
   
        # 3. evaluate the reranked data
        eval_measures = self.evaluate_reranking(reranked_run_save_path, trec_run_path=trec_run_path, qrels_path=qrels_path)
                                                        
        # 4. Save resutls to file 
        df_result = self.save_results(eval_measures, hyper_parameters, evaluation_save_path)
        print(" \n\Done evaluation ")
        print(eval_measures)

        return eval_measures



    def perform_t_test(self, baseline_run, other_runs, other_runs_names, qrels_file, query_path, eval_metrics=[MAP@5, P@1], 
                        save_path="", baseline_name="baseline", correction=None):

        # load the qrels file
        df_qrels = pd.read_csv(qrels_file, sep="\t", names=["qid", "Q0", "docno", "label"])
        df_qrels["docno"]=df_qrels["docno"].astype(str)
        df_qrels["qid"]=df_qrels["qid"].astype(str)

        # load the queries file
        df_query = Utils.read_file(query_path)
        df_query["query"] =df_query["cleaned"]
        df_query["qid"] = df_query[cf.TWEET_ID].astype(str)
        df_query = df_query[["qid", "query"]]

        runs_list = []
        runs_names = []
        # read the baseline run
        df_baseline = pd.read_csv(baseline_run, sep="\t", names=["qid", "q0", "docno", "rank", "score", "tag"])
        df_baseline["docno"]=df_baseline["docno"].astype(str)
        df_baseline["qid"]=df_baseline["qid"].astype(str)

        runs_names.append(baseline_name)
        runs_names.extend(other_runs_names)
        runs_list.append(df_baseline.copy())

        # read the runs
        for i in range(len(other_runs)):
            run_path = other_runs[i]
            df_run = pd.read_csv(run_path, sep="\t", names=["qid", "q0", "docno", "rank", "score", "tag"])
            df_run["docno"]=df_run["docno"].astype(str)
            df_run["qid"]=df_run["qid"].astype(str)

            runs_list.append(df_run.copy())

        df_eval = pt.Experiment(runs_list,
                    df_query,
                    qrels=df_qrels,
                    eval_metrics=eval_metrics,
                    names=runs_names,
                    baseline=0,
                    correction=correction,
                )
        
        if save_path != "":
            df_eval.to_excel(save_path, index=False)

        return df_eval


