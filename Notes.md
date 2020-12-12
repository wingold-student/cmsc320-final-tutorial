# Notes

Make sure to run through the Notebook file for "TODO" to see any notes/ideas/questions.

For analysis are we analyzing what makes a review negative/positive? 
* This would mean putting all reviews (of both majors) together and looking at them

We could also be looking at how reviews of a major change overtime?
* Not necessarily have to do with sentiment, but could incorporate it. Otherwise can use other variables (rating, grade)
    * Could use the sentiment learned from ALL reviews scraped. Then use it to classify even MORE reviews for us
    * See if there is a correlation between time and sentiment to a major
    * See if correlation between sentiment and a major
* Ratings / grades to time for each major

Do we want to use tags/difficulty from RMP in some way?
* Could be its own analysis

We also likely don't need to display all the data I'm printing right now in the data collection,
maybe write down/let me know which you don't think are necessary

# Setup stuff

## Environment
To get all the same dependencies/libraries I've been using, you can import
the environment yaml with conda.

If you CD into the project directory and use the command: conda activate final-tut

That should put it all together. But I may have needless libraries and such, so you can
install it all on your own if you want too.

## Viewing the database
If you want to look at stuff in the database instead of as pandas dataframes,
you can use DB Browser for SQLite. You can get it here: https://sqlitebrowser.org/dl/

## Messing with PlanetTerp and UMD.io API

If you want to play around with those API's, I'd recommend getting Postman: https://www.postman.com/downloads/

# NLP Stuff

List of possible sources for negative/positive lexicons:
* https://medium.com/@datamonsters/sentiment-analysis-tools-overview-part-1-positive-and-negative-words-databases-ae35431a470c
    * https://hlt-nlp.fbk.eu/technologies/sentiwords (-1 to 1 scoring with part of speech tag) (I've already requested the data and have it)
    * https://provalisresearch.com/products/content-analysis-software/wordstat-dictionary/sentiment-dictionaries/ (You can download just dictionary)
    * https://github.com/aesuli/SentiWordNet (Has rating too I think)
    * https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/tree/08a269765a6b185d5f3dd522c876043ba9628715/data/opinion-lexicon-English
    * https://sites.google.com/site/datascienceslab/projects/multilingualsentiment
    * http://www2.imm.dtu.dk/pubdb/pubs/6010-full.html (-5 to 5 rating)
    * https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
    * https://sentic.net/downloads/ I think we'd want to use SenticNet 1 (-1 to 1 scoring too I think)