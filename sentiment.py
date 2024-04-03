# CS331 Sentiment Analysis Assignment 3
# This file contains the processing functions

import string
from classifier import *


def process_text(text):
    """
    Preprocesses the text: Remove apostrophes, punctuation marks, etc.
    Returns a list of text
    """
    preprocessed_text = []

    for word in text:
        word = word.replace("'", "")                                      # Remove apostrophes
        word = word.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        word = word.lower()                                               # Convert to lowercase
        preprocessed_text.append(word)


    return preprocessed_text


def build_vocab(preprocessed_text):
    """
    Builds the vocab from the preprocessed text
    preprocessed_text: output from process_text
    Returns unique text tokens
    """

    vocab = []

    # 
    for sentence in preprocessed_text:
        sentence = sentence[:-1]             # Removing the class label
        tokens = sentence.split()
        vocab.extend(tokens)
       
    vocab = list(set(vocab))
    vocab.sort()


    return vocab


def vectorize_text(text, vocab):
    """
    Converts the text into vectors
    text: preprocess_text from process_text
    vocab: vocab from build_vocab
    Returns the vectorized text and the labels
    """

    vectorized_text = []
    labels = []
    m = len(vocab)

    
    for line in text:
        vector = []
        sentence, label = line.strip().split('\t')
        labels.append(int(label))
        sentence = sentence[:-1]            # Removeing class label
        
        # Iterates through the vocab array to obtain each word
        # in the vocab and checks to see if that word is in the sentence
        # if word is in the sentence then append 1 to the vector otherwise 
        # append 0 
        for word in vocab:
            if word in sentence.split():
                vector.append(1)
            else:
                vector.append(0)
        vectorized_text.append(vector)
        

    return vectorized_text, labels


def accuracy(predicted_labels, true_labels):
    """
    predicted_labels: list of 0/1s predicted by classifier
    true_labels: list of 0/1s from text file
    return the accuracy of the predictions
    """

    trueLen = len(true_labels)
    correct = 0

    #
    for value in range(trueLen):
        if predicted_labels[value] == true_labels[value]:
            correct += 1
        else:
            continue
    accuracy_score = (correct / trueLen)
    
    return accuracy_score


def main():
    # Take in text files and outputs sentiment scores

    testFile = []
    trainFile = []
    testLabel = []
    trainLabel = []

    # Read the test file data
    fileTest = open('testSet.txt', 'r')

    # Go through each line in the test file and append the sentences to
    # testFile 
    for line in fileTest:
        
        sentence = line.strip()
        testFile.append(sentence)
        
    fileTest.close

    # Call process_text and vectorize the test data
    test_text = process_text(testFile)
    testBuild_vocab = build_vocab(test_text)
    sent_test, label_test = vectorize_text(test_text, testBuild_vocab)
    
    # Read the training data file
    fileTrain = open('trainingSet.txt', 'r')
    for line in fileTrain:
        
        sentenceTrain = line.strip()
        trainFile.append(sentenceTrain)

    fileTrain.close()    

    # Call process_text and vectorize the traning data
    train_text = process_text(trainFile)
    trainBuild_vocab = build_vocab(train_text)
    vector_train, label_train = vectorize_text(train_text, trainBuild_vocab)

    # Write  the vetors for the training data to the preprocessed_train.txt 
    write_train = open('preprocessed_train.txt', 'w')
    write_train.write(','.join(trainBuild_vocab) + ',classlabel\n' )

    trainVector_size = len(vector_train)
    for x in range(trainVector_size):
        vectorTrain = vector_train[x]
        labelTrain = label_train[x]
        vector_array = [str(feature) for feature in vectorTrain]           # Create an array to hold all the vectors for each line
        write_train.write(','.join(vector_array) + ',' + str(labelTrain) + '\n')
            
    write_train.close()

    # Write the vectors for the test data to the preprocessed_test.txt file
    write_test = open('preprocessed_test.txt', 'w')
    write_test.write(','.join(testBuild_vocab) + ',classlabel\n' )

    testVector_size = len(sent_test)
    for x in range(testVector_size):
        vectorTest = sent_test[x]
        labelTest = label_test[x]
        test_vectorArray = [str(feature) for feature in vectorTest]
        write_test.write(','.join(test_vectorArray) + ',' + str(labelTest) + '\n')

    write_test.close()

    # open results and clear all the data in it
    write_results = open('results.txt', 'w')
    write_results.write('')
    write_results.close

    # Divide the training set into four equal parts 
    size_train = len(trainFile)
    numParts = 4  
    diveParts = size_train // numParts   

    for partDivide in range(numParts):

        # Starting point and ending point to get information from
        # the traningset.txt, Know where to start and when to end 
        start = 0
        end = (partDivide + 1) * diveParts
    
        # Incremental trainig base on the selected data. Start and end 
        # are the values that will determine what section of the training 
        # data will be selected  
        classifier = BayesClassifier()
        f = classifier.train(vector_train[start:end],label_train[start:end],trainBuild_vocab )

        # Testing the entire training data set
        predictions = classifier.classify_text(vector_train,trainBuild_vocab)
        acc = accuracy(predictions, label_train)
        endState = end - start
        train_results = f"Training Set Accuracy with ({endState} examples): {acc:.2%}"
        print(train_results)
    
        # Test on the test set
        predictions_test = classifier.classify_text(sent_test, testBuild_vocab)
        acc_test = accuracy(predictions_test, label_test)
        test_results = f"Test Set Accuracy: {acc_test:.2%}"
        print(test_results)

        # Write the given information to the results.txt file
        write_results = open('results.txt', 'a')
        write_results.write(f"__________________________ Part {partDivide + 1} __________________________\n\n")
        write_results.write(f"Training Set: {train_results}\n")
        write_results.write(f"Test Set: {test_results}\n")
        write_results.write('_____________________________________________________________\n\n')
        write_results.close()

    return 1


if __name__ == "__main__":
    main()



