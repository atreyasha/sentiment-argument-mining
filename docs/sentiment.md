# Sentiment Mining

## Python:

**[Lexicoder Sentiment Dictionary]("https://quanteda.io/reference/data_dictionary_LSD2015.html")**:
A lexicon designed to capture the sentiment of political texts <br>
*Possible concerns*: A good starting point, definitely more specific than other lexica but maybe rather appropriate for evaluating politics on national level. Some issues are not discussed at the international/UN level (wages, coalitions, pensions ...)

**Vader**: <br>
Lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media <br>
*Idea?*: Could we use one of the countless algorithms optimized for twitter political analysis? <br>
&rightarrow; Probably not since our texts are more sophisticated. Is it worth a try?

(*Further reading:* <https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/>)

**Textblob**:
Library built on top of nltk, brings in subjectivity as an interesting feature in addition to sentiment.
Sentiment ranges between -1.0 and 1.0, where -1.0 is the most negative, 0 is neutral and 1.0 is the most positive sentiment value. 
Subjectivity is within the range 0.0 to 1.0 where 0.0 is very objective and 1.0 is very subjective.

&rightarrow; This might be interesting for spontaneous speech, as it might be less subjective.

(*Further reading:* <https://www.paulng.me/nlp-un-general-assembly/>)

---
## R:

Three general purpose lexicons based on unigrams:

1. NRC
2. bing
3. AFINN


**NRC**: <br>
Categorizes words in a binary fashion (“yes”/“no”) into eight basic emotions (positive, negative, anger, anticipation, disgust, fear, joy, sadness, surprise, and trust)

(*Related article:* <https://medium.com/@ajgabriel_30288/sentiment-analysis-of-political-speeches-managing-unstructured-text-using-r-b090a42c0bf5>)

**Bing**: <br>
Categorizes words as positive or negative

**AFINN**: <br>
Assigns words with a score between -5 and 5, negative scores indicating negative sentiment and positive scores indicating positive sentiment


**General information on sentiment mining in R**:
<https://www.tidytextmining.com/sentiment.html>

---


### General findings:
People generally don’t seem to use special political dictionaries but just standard sentiment analysis tools as the ones mentioned above. <br>
&rightarrow; Can we improve this procedure by expanding the lexicon? <br>
&rightarrow; Is it worth creating our own lexicon?
