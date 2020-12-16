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

## Analysis / Graph ideas
* Number of reviews & professors from each major (Done)
    * I did this already, but should we try to even them out?
* Dispersion of grades reported for each major (Done?)
    * bar graph: y-axis is count and x-axis is grade. Have a column per major for each grade
* Grade dispersion over time for majors (Done)
    * With the reviews we have self-reported grades and dates of posting.
    * Y-axis could be the grade and x-axis is time
    * Could separate into years or month/year or semester using the "date" column
* Rating dispersion over time for majors (Done?)
    * Similar to grade dispersion
* Sentiment dispersion over tiem for majors
    * Again, similar to grade dispersion
* Look at how many positive/negative word in relation to total words
    * Doing this per major, not the most exciting graph though
* Some mapping/graph of negative words to rating given AND/OR grade received 
* Look at tag data of reviews or professors from RateMyProfessor

## Sentiment Stuff
I just did as the TA suggested and add/subtracted a score based on negative and positive words.
So the 'sentiment_ground_label' is the positive/negative/neutral label, where
1 is positive, -1 is negative, and 0 is neutral.

I also have a function that can normalize the scoring between -1 and 1, so that we could
gauge how negative or positive the review is. It is a VERY simple swap, so let me know if you
want to try that instead.

## ML Ideas
* Use the sentiment labels for reviews to train for detecting a negative/positive/neutral review
* Linear/logistic regression (and/or gradient descent) for predicting future semesters
    * This could be grade, sentiment, rating, etc.
* Look to see if there is a tie between the review body (sentiment) and the grade/rating/difficulty
    * Though I think this may need larger scale than just -1, 0, or 1 range for sentiment

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

Good resource: https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/

I currently chose to use: https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon

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