Reviews Caregorization
======================

![Python 3.9.13](https://img.shields.io/badge/python-3.9.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-ref.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15.9-green.svg)
![pandas](https://img.shields.io/badge/pandas-black.svg)
![numpy](https://img.shields.io/badge/numpy-blue.svg)
![gensim](https://img.shields.io/badge/gensim-black.svg)
![sklearn](https://img.shields.io/badge/sklearn-orange.svg)
![matplotlib](https://img.shields.io/badge/matplotlib-black.svg)
![EasyNMT](https://img.shields.io/badge/EasyNMT-red.svg)


Categorization of customer feedback is used by various companies to improve the quality of the product or service they offer. This program deals with language-independent categorization of customer feedback. Conversion between languages uses vector space transformation using a transformation matrix and machine translation. The data corpus for training and testing classifiers is created from reviews of the chain McDonald’s, in which the sentiment in selected categories is then manually annotated. In this way, the training corpus is created from Czech reviews and the test corpus from English and German reviews. 



## Instalation

The file with the program contains the text file requirements.txt, which contains the necessary libraries for the correct operation of the program, including the recommended versions. These libraries can be installed using the command:

    pip install -r requirements.txt

For a machine with an NVIDIA GPU, it is possible to install the torch library with the –index-url parameter https://download.pytorch.org/whl/cu117 so that the program can run on the GPU and perform parallel calculations:

    pip install torch --index-url https://download.pytorch.org/whl/cu117

## Run

The program has two parts - a console application and a demonstrator. 

### Console application

The console application can be started by entering the command python main.py and several required and optional parameters:
   - **–a (action)** – Indicates what action the program should perform. This parameter is always mandatory and makes the other parameters mandatory.
       - 'mono' – defines monolingual classification
        ’cross’ – defines multilingual classification using a transformation matrix
       - ’translate’ – defines a multilingual classification using translation
       - ’model’ – defines the creation of a model
      
   - **–mt (model type)** – Specifies the type of vector models. This parameter is always mandatory.
     - ’ft’ – defines the fasttext model type
     - ’w2v’ – defines the word2vec model type
     - ’tfidf’ – defines the tf-idf model type
     - ’bow’ – defines the bow model type
    
   - **–mp (model path)** – Specifies the path to the file name containing the vector model for the language to be trained. This parameter is mandatory if the model type is word2vec or fasttext.
            
   - **–mptest (model path test)** – Specifies the path to the file name containing the vector model for the language to be tested. This parameter is mandatory if the program action is ’model’, ’cross’ or ’translate’ and the vector model type is word2vec or fasttext.
            
   - **–cm (model type)** – Specifies the type of classification models. This parameter is always required.
     - ’lstm’ – defines the fasttext model type
     - ’cnn’ – defines the word2vec model type
     - ’svm’ – defines the Support vector machines model type
     - ’logreg’ – defines the Logistic regression model type
     - ’dectree’ – defines the type of Decision tree model
    
   - **–rp (reviews path)** – Specifies the path to the file name containing the training reviews. This parameter is mandatory if the program action is 'mono', 'cross' or 'translate'.
            
   - **–rptest (reviews path test)** – Specifies the path to the file name containing the reviews for testing. This parameter is mandatory if the program action is 'cross' or 'translate'.
            
   - **–l (language)** – Specifies the language of the training data. Text preprocessing, translation and dictionary for calculating the transformation matrix depend on the language. This parameter is always mandatory.
            
   - **–ltest (language test)** – Specifies the language of the test data. This parameter is mandatory if the program action is 'cross' or 'translate'.
            
   - **–fp (feed path)** – Specifies the path to the file name containing the reviews to create the vector model. This parameter is mandatory if the program action is 'model'.
            
                        
Neural network classifiers (*lstm, cnn*) can only be combined with *word2vec* and *fasttext* text representation, and other classifiers (*Support vector machines, Logistic regression* and *Decision tree*) can only be combined with *bow* and *tf-idf* text representation. Multilingual transformation can only be used with neural network classifiers (*lstm, cnn*). After the completion of the program run, the results are displayed in the console itself and in the text file *log.txt*.


### Examples of use

Examples of execution based on what is expected of the program:

- To create a word2vec vector model for the Czech language from the data set feed_cs.xlsx, the command is run:
    
     python main.py -a model -mp data/vec_model/w2v_cs.bin -l cs -fp data/feed/feed_cs.xlsx -mt w2v
  
  
- For monolingual classification with the tf-idf vector model and the svm classifier for the Czech language, the command is run:

     python main.py -a mono -rp data/review/reviews_cs.xlsx -l cs -mt tfidf -cm svm


- For multilingual classification using the transformation matrix with the word2vec vector model and the lstm classifier for Czech training data and German test data, the command is run:

     python main.py -a cross -rp data/review/reviews_cs.xlsx -mp data/vec_model/w2v_cs.bin -l cs -rptest data/ review/reviews_de.xlsx -mptest data/vec_model/w2v_de.bin -ltest de -mt w2v -cm lstm


- For multilingual classification using translation with fasttext vector model and lstm classifier for Czech training data and German test data, the command is run:

     python main.py -a translate -rp data/review/reviews_cs.xlsx -mp data/vec_model/ft_cs.bin -l cs -rptest data/ review/reviews_de.xlsx -ltest de -mt ft -cm lstm


### Application demonstrator

The program expects a fasttext vector model for each language stored in the file: data/vec_model/ft_cs.bin, analogously en or de for English and German. So the models must be pre-created by the console application. 

- The application demonstrator is started with the command:

     python gui.py
     