# Sentiment-Analysis-From-Product-Review 
F o r  L 6 4 5 / B 6 5 9 A d v a n c e d N a t u r a l L a n g u a g e P r o c e s s i n g 

INSTRUCTION SET 

Requirements  
Python 3.5  
Cython  
NLTK 
Gensim 
Scikit-learn 0.19 
pandas 0.19 
matplotlib 
Numpy 
Scipy 
  
The program files can be imported to your favorite python editor( in my case it was pycharm) and can be run.
 
Each file can be run individually as a main method has been provided for that purpose. Files that can be run individually are mentioned below: -  
FetchData.py :- This takes a text file as input and converts it into CSV file for further processing. This file needs to be run first.  
NaiveBayes.py :- This file provides us with default naïve bayes and optimized naïve bayes results. 
BagOfWords.py :- This file provides us with the result of random forest classifier combined with bag of words model. 
Word2Vecor.py :- This file applies the deep learning neural net provided by gensim to out dataset and provides us the result. 
Doc2Vecor.py :- This file applies the improved word2vector model where the context is preserved. 
Create_TIMBL.py :-  This file creates the input TIMBL file. The output is created in two folders named “TIMBL without Subjectivity Lexicon” and “TIMBL with Subjectivity Lexicon”. Each folder will create a test and a train file for out experiment. This file is then taken and run on KARST (IU’s supercomputer that has TIMBL installed). The command used to run it on KARST is :-  
	
	>Timbl -f trainfile -t testfile 
	> default run 
