For the Gutenberg Library the files under the directory - have some files contains the author name and title:
-/auth_*_lang_en_by_popularity.js
var json_data = [["Moby Dick; Or, The Whale", "Herman Melville", "110", 2701], ["Bartleby, the Scrivener: A Story of Wall-Street", "Herman Melville", "110", 11231]

It is an array of array of [title, author, <not sure>, gutenberg ebook identifer]
https://www.gutenberg.org/ebooks/3203

If I want to use Gutenberg books here's a strategy (and some pain points):
- I cannot extract all the files without almost completely filling up my computer's disk space, 50gb is just way way too big
- consider using this gutenberg library as a side project, for older text, I don't think i would even be able to tokenize or analyze all the data, my ram is not enough to hold all the data.

## Project requirements
1. 500 kindle books with commonly read books well be used for the unsupervised portion of the project
2. Gutenberg library books which contain lots of types of book that are uninteresting to me will also be the unsupervised portion of the project:
    - how to exclude dictionaries
    - how to exclude encyclopedias
3. Using recommendation data from Kaggle I can also create a recommendation system, which...should contain some supervised learning? I am not sure about this part of the project.

I can display four things:
1. top rankings for a cluster (not genre) based on ratings (which changes over time if there are more data / cleaning / modeling params)
2. top rankings for a cluster based on cosine similarity
3. books within a cluster compared to a target book sorted by cosine sim
4. books within a cluster compared to a target book sorted by ratings
5. mean and std of scores for clusters
6. top 3 genres for clusters (requires good reads metadata)
7. what clustering looks like when I am using a metric such as:
    - minimizing count of <series> clusters * <cosine_sim of those books> e.g.:
        - number of HP clusters * sum of cosine_sim with harry potter books
        - number of hitchhiker clusters * sum of cosine_sim with hitchiker books
        - number of got clusters * sum of cosine_sim with got books
        - number of orson scott card clusters * sum of cosine_sim with got books
        - number of frank herbert clusters * sum of cosine_sim with got books
        - number of steig larson girl who played with fire clusters * sum of cosine_sim with got books
        - number of Lewis CS who played with fire clusters * sum of cosine_sim with got books
        - number of suzanne collins who played with fire clusters * sum of cosine_sim with got books
    
8. goodreads to get ratings and review count (and existence) and decide on whether to include a corpus in my database.

### Book recommendation system (I will be doing this)
MVP - kindle books convert to a .txt file, can probably discard the first ~500 lines (foreword, table of contents, publishers information) and the last ~200 lines.

1. Feature engineering
    - I need to clean the data or something first. But I think that's part of the tokenizing process?
    - Read 1-2 articles on how to clean book data for NLP analysis
    - cleaning will be a pipeline (or return a pipeline) that takes a configuration, e.g. try removing nouns, try removing punctuation, lemming, etc.
    - This pipeline will create data points that can then be used for different types of analysis, the most basic will be e.g. Tokenize --> TF-IDF --> NMF --> k-means
    - a more complicated analysis could be: Tokenize specifically for dialogue --> TF-IDF --> Sentiment Analysis --> k-means
        - would need to read 1-2 articles on NLP and sentiment analysis.
    

2. run tokenizer
    - CountVectorizer
    - TF-IDF Vectorizer
3. reduce dimensions
    - LSA
    - LDA
    - NMF
4. Cluster models
    - k-means (can predict)
    - DBSCAN
    - Hierarchal (cannot predict only classify?)
    - ...
5. given a book, return a list of the most similar books
6. After clustering t-SNE can be used to visualize
7. Look into and understand more Bayes Optimization and see if it can used instead of GridSearchCV.

Nice to haves:
1. larger corpus
    - If had meta information on the books, such as the genre and year then...maybe I could create different models for different periods. Though I don't know if that's desirable. For example if you like a modern 21st century book, do you care about a 18th century book? And vice versa. Contemporary vs non-contemporary, maybe this is something the ML algo can do on its own.
2. try different tokenizers
3. cross reference returned books with good-reads
4. download word2vec...or something
5. more "cleaning" and feature engineering on the corpus themselves
    - maybe slices of text is good enough, for example if a book is good enough to finish, then maybe the last 30% of the book is enough
6. Is it possible to parallel process these modeling?


### News recommendation
For news recommendation...it's kinda like who cares, once I read a piece of news I don't need to read multiple news sources on the same subject. It's likely just reading one source is good enough to give me the gist of the issue. Probably more useful for research.

For book character profiles
- How can I extract character dialogues from a book, that in and of itself seems like a daunting task, this seems like:
1. a deterministic challenge. Can probably hardcode a lot of text extractions
2. a supervised learning model. extracting text and seeing if text is correctly extracted.
3.


### Can I read text and have it play a video for me?
- this would be way too ambitious i believe, but i like the idea because maybe if you write down something, like a play/script then it can be during into something visual
it reminds me of the episode of star trek where teammates were kidnapped by aliens and had their memories altered.
what would a basic verion of this look like?
- Read a sentence
- maybe have a database of nouns, actions, places, etc.
- but this feels like way too much time would be spent on the animation engine
- More like, what's the data science part, the unsupervised learning part?
