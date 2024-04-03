# This file implements a Naive Bayes Classifier
import math


class BayesClassifier():
    """
    Naive Bayes Classifier
    file length: file length of training file
    sections: sections for incremental training
    """
    def __init__(self):
        self.postive_word_counts = {}
        self.negative_word_counts = {}
        self.percent_positive_scentences = 0
        self.percent_negative_scentences = 0
        self.file_length = 499
        self.file_sections = [self.file_length // 4, self.file_length // 3, self.file_length // 2]
        self.vocabulary = set()

    def train(self, train_data, train_labels, vocab):
        """
        This function builds the word counts and sentence percentages used for classify_text
        train_data: vectorized text
        train_labels: vectorized labels
        vocab: vocab from build_vocab
        """

        train_size = len(train_data)

        # Iterates over the indices of the train_data allowing us to
        # access each element in train_data and train_label
        for x in range(train_size):

            # Assigns the x-th element of train_data to the vairable
            # represents the feature vector for a particualr training example
            vector = train_data[x]
            label = train_labels[x]                 # Represents the label (0 or 1),for the corresponding train example

            # Checks if the labe == 1, if so increment 1 to percent_positive_scentence.
            # Indicating a positive sentiment
            if label == 1:
                self.percent_positive_scentences += 1
            else:
                self.percent_negative_scentences += 1

            vocab_size = len(vocab)


            # Iterates over the indices of the vocab list.
            for j in range(vocab_size):

                # Checks if the j-th element of the feature vector is 1, 
                # this will indicate the presence of the corresponding word in the sentence
                if vector[j] == 1:
                    word = vocab[j]
                    self.vocabulary.add(word)

                    if label == 1:
                        # Increments the count of the word in the positive sentiment class
                        # the method get is used to retrieve the current count of the word and if it does
                        # not exist then return the value 0
                        self.postive_word_counts[word] = self.postive_word_counts.get(word, 0) + 1
                    else:
                        # Increments the count of the word in the negative sentiment class
                        # the method get is used to retrieve the current count of the word and if it does
                        # not exist then return the value 0
                        self.negative_word_counts[word] = self.negative_word_counts.get(word, 0) + 1

            # Incremental training based on file sections
            
            if x + 1 in self.file_sections:
                
                self.file_sections.remove(x + 1)
                self.perform_incremental_training(train_data[:x + 1], train_labels[:x + 1], vocab)

       
    

    def perform_incremental_training(self, train_data, train_labels, vocab):
        """
        Perform incremental training on the given section of data
        train_data: vectorized text
        train_labels: vectorized labels
        vocab: vocab from build_vocab
        """
        train_size = len(train_data)

        for x in range(train_size):
            vector = train_data[x]
            label = train_labels[x]

            if label == 1:
                self.percent_positive_scentences += 1
            else:
                self.percent_negative_scentences += 1

            vocab_size = len(vocab)

            for j in range(vocab_size):
                if vector[j] == 1:
                    word = vocab[j]
                    self.vocabulary.add(word)

                    if label == 1:
                        self.postive_word_counts[word] = self.postive_word_counts.get(word, 0) + 1
                    else:
                        self.negative_word_counts[word] = self.negative_word_counts.get(word, 0) + 1

        return 1


    def classify_text(self, vectors, vocab):
        """
        vectors: [vector1, vector2, ...]
        predictions: [0, 1, ...]
        """

        predictions = []
        for vector in vectors:
            positive_score = 0
            negative_score = 0
            for i, word in enumerate(vocab):
                if vector[i] == 1:
                    positive_score += math.log((self.postive_word_counts.get(word, 0) + 1) / (self.percent_positive_scentences + len(vocab)))
                    negative_score += math.log((self.negative_word_counts.get(word, 0) + 1) / (self.percent_negative_scentences + len(vocab)))
            if positive_score >= negative_score:
                predictions.append(1)
            else:
                predictions.append(0)
        


        return predictions
    

    
    