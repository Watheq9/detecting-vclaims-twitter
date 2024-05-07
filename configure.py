
ENG_CLEF_2020_QRELS = "./data/CLEF_2020/dataset/tweet-vclaim-pairs.qrels"
ENG_CLEF_2020_TRAIN_QUERIES = "./data/CLEF_2020/dataset/train-tweets.queries.tsv"
ENG_CLEF_2020_VCLAIMS = "./data/CLEF_2020/dataset/verified_claims.docs.tsv"
ENG_CLEF_2020_DEV_QUERIES  = "./data/CLEF_2020/dataset/dev-tweets-queries.tsv"
ENG_CLEF_2020_TEST_QUERIES  = "./data/CLEF_2020/dataset/test-tweets.queries.tsv"

ENG_CLEF_2020_URL_CLEANED_TRAIN_QUERIES = "./data/CLEF_2020/url_cleaned_train_queries.xlsx"
ENG_CLEF_2020_URL_CLEANED_DEV_QUERIES = "./data/CLEF_2020/url_cleaned_dev_queries.xlsx"
ENG_CLEF_2020_URL_CLEANED_TEST_QUERIES = "./data/CLEF_2020/url_cleaned_test_queries.xlsx"


AR_2021_QRElS_FILE = "./data/CLEF_2021/Arabic/dataset/CT2021-Task2A-AR-ALL_QRELs.txt"
AR_2021_VCLAIMS_FILE = "./data/CLEF_2021/Arabic/dataset/CT2021-Task2A-AR-Verified_Claims.txt"
AR_2021_ALL_QUERIES_PATH = "./data/CLEF_2021/Arabic/dataset/CT2021-Task2A-AR-ALL_Queries.txt"  # all = train + dev
AR_2021_DEV_QUERIES_PATH = "./data/CLEF_2021/Arabic/dataset/CT2021-Task2A-AR-Dev_Queries.txt"
AR_2021_TRAIN_QUERIES_PATH = "./data/CLEF_2021/Arabic/dataset/CT2021-Task2A-AR-Train_Queries.txt"
AR_2021_TEST_QUERIES_PATH = "./data/CLEF_2021/Arabic/dataset/CT2021-Task2A-AR-Test_Queries.tsv"
AR_2021_EVALUATION_FILE = "./data/ar_clef2021_evaluation.xlsx"

AR_2021_URL_CLEANED_TRAIN_QUERIES  = "./data/CLEF_2021/Arabic/dataset/url_cleaned_train_queries.xlsx"
AR_2021_URL_CLEANED_DEV_QUERIES  = "./data/CLEF_2021/Arabic/dataset/url_cleaned_dev_queries.xlsx"
AR_2021_URL_CLEANED_TEST_QUERIES  = "./data/CLEF_2021/Arabic/dataset/url_cleaned_test_queries.xlsx"



ENG_2021_QRELS = "./data/CLEF_2021/English/dataset/EN-all_QRELs.txt"
ENG_2021_ALL_QUERIES = "./data/CLEF_2021/English/dataset/EN_all_queries.tsv"
ENG_2021_TRAIN_QUERIES = "./data/CLEF_2021/English/dataset/EN_train_queries.tsv"
ENG_2021_DEV_QUERIES = "./data/CLEF_2021/English/dataset/EN_dev_queries.tsv"
ENG_2021_TEST_QUERIES = "./data/CLEF_2021/English/dataset/EN_test_queries.tsv"
ENG_2021_VCLAIMS = "./data/CLEF_2021/English/dataset/CT2021-Task2A-EN-Verified_Claims.txt"

ENG_CLEF_2021_URL_CLEANED_TRAIN_QUERIES  = "./data/CLEF_2021/English/url_cleaned_train_queries_2021.xlsx"
ENG_CLEF_2021_URL_CLEANED_DEV_QUERIES = "./data/CLEF_2021/English/url_cleaned_dev_queries_2021.xlsx"
ENG_CLEF_2021_URL_CLEANED_TEST_QUERIES  = "./data/CLEF_2021/English/url_cleaned_test_queries_2021.xlsx"


TWEET_ID_COLUMN = "tweet_id"
TWEET_TEXT_COLUMN = "tweet_text"
TWEET_ID = "tweet_id"
TWEET_TEXT = "tweet_text"
CONTENTS = "contents"
TITLE = "title"
CLEANED = "cleaned"

ENG_CLEF_2020_EVALUATION_FILE = "./data/eng_clef_2020_evaluation.xlsx"
ENGLISH_QUERIES = "english_queries"

CLAIM_ID = "claim_id"
VCLAIM_ID = "vclaim_id"
VCLAIM = "vclaim"
CLAIM_TEXT = "contents"
LABEL = "label"
RANK = "rank"
TAG = "tag"
SCORE = "score"
TWEET_EMBEDDING = "tweet_embedding"
CLAIM_EMBEDDING = "claim_embedding"
SIMILARITY_SCORE = "cosine_similarity_score"
COMMON_KEYWORDS = "common keywords"


Q0 = "Q0"
TEXT = "text"
QUERY = "query"
QUERY_ID = "query_id"
QID = "qid"
DOC_NO = "docno"
DOCID = "docid"
FIRST_DOCUMENT = "first_document"
FIRST_DOCUMENT_ID = "first_document_id"
FIRST_DOCUMENT_RANK = "first_document_rank"
SECOND_DOCUMENT = "second_document"
SECOND_DOCUMENT_ID = "second_document_id"
SECOND_DOCUMENT_RANK = "second_document_rank"
DOCUMENT_ID = "document_id"
ONE_LAYER = 1
TWO_LAYERS = 2

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

LEXICAL_SIMILARITY = "lexical_similarity"
SEMANTIC_SIMILARITY = "semantic_similarity"
NORMAL_CURRICULA = 1
ALL_RELEVANT_CURRICULA = 2
ALL_NON_RELEVANT_CURRICULA = 3
HARMONIC_MEAN_OF_SIMILARITY = 4
HARMONIC_MEAN_OF_RECIP_RANK_AND_SIMILARITY = 5

QUERY_AND_TITLE_INPUT_IDS = "QUERY_AND_TITLE_INPUT_IDS"
QUERY_AND_TITLE_ATTENTION_MASK = "QUERY_AND_TITLE_ATTENTION_MASK"

VCLAIM_ONLY= "VCLAIM_ONLY"
TITLE_ONLY= "TITLE_ONLY"
VCLAIM_AND_TITLE = "VCLAIM_AND_TITLE"

EPSILON = 1e-8
EVALUATION_PATH = "evaluation_path"
RELEVANCE_THRESHOLD = "RELEVANCE_THRESHOLD"
RERANKED_PAIRS_PATH = "reranked_pairs_run_path"
EVALUATION_TYPE = "EVALUATION_TYPE"
DEVELOPEMENT = "DEVELOPEMENT"
TESTING = "TESTING"

INITIAL_RETRIEVAL_RUN = "INITIAL_RETRIEVAL_RUN"

MONO_PAIRS_PATH = "MONO_PAIRS_PATH"
DEV_MONO_PAIRS_PATH = "DEV_MONO_PAIRS_PATH"
TEST_MONO_PAIRS_PATH = "TEST_MONO_PAIRS_PATH"
LANG = "LANG"

WHAT_TO_TEST = "WHAT_TO_TEST"
QRELS_PATH = "QRELS_PATH"
VCLAIMS_PATH = "VCLAIMS_PATH"
INDEX_PATH = "INDEX_PATH"
TRAIN_QUERY = "TRAIN_QUERY"
DEV_QUERY = "DEV_QUERY"
TEST_QUERY = "TEST_QUERY"
BM25_TRAIN_RUN = "BM25_TRAIN_RUN"
BM25_DEV_RUN = "BM25_DEV_RUN"
BM25_TEST_RUN = "BM25_TEST_RUN"
QUERY_COLUMN = "QUERY_COLUMN"
TRAIN_SET = "TRAIN_SET"
DEV_SET = "DEV_SET"
TEST_SET = "TEST_SET"
DEPTH_OF_RANDOM = "DEPTH_OF_RANDOM"

