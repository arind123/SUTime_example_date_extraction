# SUTime_example_date_extraction
Using Stanford sutime python wrapper for date extraction from articles. Then get the sentence with the date extract for relevent sentence extraction from a news article. 
For the sentence with the date for the summarization Result and extracted subject verb object phrase
, which can be optimized as per requiremnt (For this part I took the codes from Rock de Vocht(peter3125)(https://github.com/peter3125/enhanced-subject-verb-
object-extraction and some adjustments in the "dependency markers for subjects and objects" refered from the artice https://suttipong-kull.medium.com/how-to-extract-subject-** verb-and-object-by-nlp-4149323a7d7d) :

DATED: 12/10/2021
* We are now first taking each sentence of a paragraph and then searching dates and SVOs, inplace of our previous approach of first findeing dates and then getting corresponding sentences(Makes life a lot easier with respect to the coding part).
* We have excluded those dates for which the taged "value" starts with "P"(e.g. P1D,P5M etc) as those are not really making much sense for our context.
* Need reefinement of the SVO technique for better result.
