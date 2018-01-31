from collections import Counter
import re
import io
import string
import math, collections
import operator
from nltk.stem.snowball import *
import string

######### NOTE: PUNCTUATION IS NOT INCLUDED. THIS IS THE ONLY WAY THE ACCURACY WENT AT OR ABOVE 50% (Otherwise, it was at 48.5%) ##########

###################################### Problem 1 (TEXT CLASSIFICATION) #########################################################################################################################

totalNumberOfDocs = 0 #HOLDS THE TOTAL NUMBER OF LINES
totalUniqueWords = 0 #HOLDS NUMBER OF UNIQUE WORDS STATED BY EVERYONE ALTOGETHER
listOfUniqueWords = {} #Holds the unique words of ALL the speakers COMBINED
docsPerClass = {} #Holds how many lines each person has spoken
wordCountsPerSpeaker = {} #Holds how many times each speaker has said a word from all of the possible words (Add-One smoothing is applied)
totalWordsPerSpeaker = {} #Holds counts for total words per speaker (with unique vocabulary of ALL speakers added in)

file = open('train', 'r') #Opens the training file
for line in file:
    lineTokenized = line.split() #Splits the current line into tokens
    currentSpeaker = lineTokenized[0] #Holds the speaker of the current line
    currentSpeakerWords = [lineTokenized[x] for x in range(1,len(lineTokenized)) if lineTokenized[x] not in string.punctuation] #Holds a list of the words [DOES NOT INCLUDE PUNCTUATION] said by the speaker of the current line (in order)

    if(currentSpeaker not in docsPerClass): #If the current speaker is not in docsPerClass
        docsPerClass[currentSpeaker] = 1 #Initialize
    else:
        docsPerClass[currentSpeaker] += 1 #Add 1 to the number of documents for the current speaker
    totalNumberOfDocs += 1 #Increases the number of documents regardless

    if(currentSpeaker not in wordCountsPerSpeaker): #If the current speaker is not in wordCountsPerSpeaker
        wordCountsPerSpeaker[currentSpeaker] = collections.defaultdict(lambda: 1) #Add-One smoothing to account for words each speaker has not said
        for i in range(len(currentSpeakerWords)):
            token = currentSpeakerWords[i] #Current word in the list
            wordCountsPerSpeaker[currentSpeaker][token] = wordCountsPerSpeaker[currentSpeaker][token] + 1 #The count for the current word in the list is increased by 1 for the current speaker
    else:
        for i in range(len(currentSpeakerWords)):
            token = currentSpeakerWords[i] #Current word in the list
            wordCountsPerSpeaker[currentSpeaker][token] = wordCountsPerSpeaker[currentSpeaker][token] + 1 #The count for the current word in the list is increased by 1 for the current speaker

for key, value in wordCountsPerSpeaker.items():
    speaker = key
    totalWordsPerSpeaker[speaker] = 0 #No need to check if speaker is already in list because wordCountsPerSpeaker has each speaker listed once
    for key, value in wordCountsPerSpeaker[speaker].items(): #Go through all the words of each speaker
        currentWord = key
        totalWordsPerSpeaker[speaker] += value #Increase that speakers total word count accordingly

        if(currentWord not in listOfUniqueWords): #If a word said by a given speaker is not in the list of unique words
            listOfUniqueWords[currentWord] = 1

for key, value in listOfUniqueWords.items(): #Counting up the total number of unique words used by ALL the speakers combined
    totalUniqueWords += 1

for key, value in totalWordsPerSpeaker.items():
    totalWordsPerSpeaker[key] += totalUniqueWords #This is used for Add-One smoothing, in which the unique vocabulary of ALL the speakers is added

################### DEV SET RESULTS ##########################################################################################################################################################
devAccuracyList = [] #Will hold 1 for correct prediction of a line's class(speaker), and 0 for incorrect prediction of a line's class(speaker)

devFile = open('dev', 'r') #Dev set
for line in devFile:
    lineTokenized = line.split() #Splits the current line into tokens
    currentSpeaker = lineTokenized[0] #Holds the speaker of the current line
    currentSpeakerWords = [lineTokenized[x] for x in range(1,len(lineTokenized)) if lineTokenized[x] not in string.punctuation] #Holds a list of the words [DOES NOT INCLUDE PUNCTUATION] said by the speaker of the current line (in order)

    classLikelihoodList = {} #Holds likelihood each speaker said the current line
    for key, value in totalWordsPerSpeaker.items(): #Go through each speaker from the training set
        tempSpeaker = key
        tempProbability = 0.0 #Initialize the probability of the current speaker having said the current line
        prior = math.log( docsPerClass[tempSpeaker] / float(totalNumberOfDocs) ) #Likelihood of the class(speaker) [a.k.a the prior]
        tempProbability += prior #Add the prior to the probability

        for i in range(len(currentSpeakerWords)): #Go through each word that the current speaker is saying
            tempProbability += math.log( float(wordCountsPerSpeaker[tempSpeaker][currentSpeakerWords[i]]) / totalWordsPerSpeaker[tempSpeaker] ) #Add probability of word given class (speaker)

        if(tempSpeaker not in classLikelihoodList):
            classLikelihoodList[tempSpeaker] = tempProbability #Initializes probability of each speaker to 1

    predictedSpeaker = max(classLikelihoodList, key = classLikelihoodList.get) #Sets prediction equal to the speaker with the highest likelihood for the line

    if(predictedSpeaker == currentSpeaker): #Append a 1 for a correct prediction
        devAccuracyList.append(1)
    else:                                   #Append a 0 for an incorrect prediction
        devAccuracyList.append(0)

print("Accuracy of model on dev set: ", devAccuracyList.count(1) / len(devAccuracyList)) #Prints Accuracy of Model on Dev set (Number of correct predictions / Total Number of Predictions)
###END OF DEV SET RESULTS

#################### TEST SET RESULTS ###########################################################################################################################################################
testAccuracyList = [] #Will hold 1 for correct prediction of a line's class(speaker), and 0 for incorrect prediction of a line's class(speaker)

testFile = open('test', 'r') #Test set
for line in testFile:
    lineTokenized = line.split() #Splits the current line into tokens
    currentSpeaker = lineTokenized[0] #Holds the speaker of the current line
    currentSpeakerWords = [lineTokenized[x] for x in range(1,len(lineTokenized)) if lineTokenized[x] not in string.punctuation] #Holds a list of the words [DOES NOT INCLUDE PUNCTUATION] said by the speaker of the current line (in order)

    classLikelihoodList = {} #Holds likelihood each speaker said the current line
    for key, value in totalWordsPerSpeaker.items(): #Go through each speaker from the training set
        tempSpeaker = key
        tempProbability = 0.0 #Initialize the probability of the current speaker having said the current line
        prior = math.log( docsPerClass[tempSpeaker] / float(totalNumberOfDocs) ) #Likelihood of the class(speaker) in general
        tempProbability += prior #Add the prior to the probability

        for i in range(len(currentSpeakerWords)): #Go through each word that the current speaker is saying
            tempProbability += math.log( float(wordCountsPerSpeaker[tempSpeaker][currentSpeakerWords[i]]) / totalWordsPerSpeaker[tempSpeaker] ) #Add probability of word given class (speaker)

        if(tempSpeaker not in classLikelihoodList):
            classLikelihoodList[tempSpeaker] = tempProbability #Initializes probability of each speaker to 1

    predictedSpeaker = max(classLikelihoodList, key = classLikelihoodList.get) #Sets prediction equal to the speaker with the highest likelihood for the line

    if(predictedSpeaker == currentSpeaker): #Append a 1 for a correct prediction
        testAccuracyList.append(1)
    else:                                   #Append a 0 for an incorrect prediction
        testAccuracyList.append(0)

print("Accuracy of model on test set: ", testAccuracyList.count(1) / len(testAccuracyList)) #Prints Accuracy of Model on Test set (Number of correct predictions / Total Number of Predictions)
###END OF TEST SET RESULTS










###################################### Problem 2 (FEATURE ENGINEERING) ###########################################################################################################################
############################################# (Part A) BIGRAMS TESTING #################################################################################
totalNumberOfDocs = 0 #HOLDS THE TOTAL NUMBER OF LINES
totalUniqueBigrams = 0 #HOLDS NUMBER OF UNIQUE BIGRAMS STATED BY EVERYONE ALTOGETHER
listOfUniqueBigrams = {} #Holds the unique bigrams of ALL the speakers COMBINED
docsPerClass = {} #Holds how many lines each person has spoken
bigramCountsPerSpeaker = {} #Holds how many times each speaker has said a bigram from all of the possible bigrams (Add-One smoothing is applied)
totalBigramsPerSpeaker = {} #Holds counts for total bigrams per speaker (with unique vocabulary of ALL speakers added in)

file = open('train', 'r') #Opens the training file
for line in file:
    lineTokenized = line.split() #Splits the current line into tokens
    currentSpeaker = lineTokenized[0] #Holds the speaker of the current line
    currentSpeakerWords = [lineTokenized[x] for x in range(1,len(lineTokenized)) if lineTokenized[x] not in string.punctuation] #Holds a list of the words [DOES NOT INCLUDE PUNCTUATION] said by the speaker of the current line (in order)

    if(currentSpeaker not in docsPerClass): #If the current speaker is not in docsPerClass
        docsPerClass[currentSpeaker] = 1 #Initialize
    else:
        docsPerClass[currentSpeaker] += 1 #Add 1 to the number of documents for the current speaker
    totalNumberOfDocs += 1 #Increases the number of documents regardless

    #BUILDING THE BIGRAM MODEL
    if (currentSpeaker not in bigramCountsPerSpeaker):  # If the current speaker is not in bigramCountsPerSpeaker
        bigramCountsPerSpeaker[currentSpeaker] = collections.defaultdict(lambda: 1)  # Add-One smoothing to account for bigrams each speaker has not said
        for i in range(len(currentSpeakerWords)):
            if(i != 0): #General rules; If not at the first word on the line
                bigramCountsPerSpeaker[currentSpeaker][ (currentSpeakerWords[i-1], currentSpeakerWords[i]) ] += 1
            elif(i == 0):
                bigramCountsPerSpeaker[currentSpeaker][ ('<s>', currentSpeakerWords[i]) ] += 1
            #A WORD MAY SATISFY BOTH THE FIRST BIGRAM CONDITIONAL AND THE ONE BELOW
            if(i == len(currentSpeakerWords) - 1):
                bigramCountsPerSpeaker[currentSpeaker][ (currentSpeakerWords[i], '</s>') ] += 1
    else:
        for i in range(len(currentSpeakerWords)):
            if(i != 0): #General rules; If not at the first word on the line
                bigramCountsPerSpeaker[currentSpeaker][ (currentSpeakerWords[i-1], currentSpeakerWords[i]) ] += 1
            elif(i == 0): #If the first word in the line
                bigramCountsPerSpeaker[currentSpeaker][ ('<s>', currentSpeakerWords[i]) ] += 1
            #A WORD MAY SATISFY BOTH THE FIRST BIGRAM CONDITIONAL AND THE ONE BELOW
            if(i == len(currentSpeakerWords) - 1): #If the last word in the line
                bigramCountsPerSpeaker[currentSpeaker][ (currentSpeakerWords[i], '</s>') ] += 1

for key, value in bigramCountsPerSpeaker.items():
    speaker = key
    totalBigramsPerSpeaker[speaker] = 0 #No need to check if speaker is already in list because bigramCountsPerSpeaker has each speaker listed once
    for key, value in bigramCountsPerSpeaker[speaker].items(): #Go through all the bigrams of each speaker
        currentBigram = key
        totalBigramsPerSpeaker[speaker] += value #Increase that speakers total bigram count accordingly

        if(currentBigram not in listOfUniqueBigrams): #If a bigram said by a given speaker is not in the list of unique bigrams
            listOfUniqueBigrams[currentBigram] = 1

for key, value in listOfUniqueBigrams.items(): #Counting up the total number of unique bigrams used by ALL the speakers combined
    totalUniqueBigrams += 1

for key, value in totalBigramsPerSpeaker.items():
    totalBigramsPerSpeaker[key] += totalUniqueBigrams #This is used for Add-One smoothing, in which the unique bigrams of ALL the speakers is added

################### DEV SET RESULTS ##########################################################################################################################################################
devAccuracyList = [] #Will hold 1 for correct prediction of a line's class(speaker), and 0 for incorrect prediction of a line's class(speaker)

devFile = open('dev', 'r') #Dev set
for line in devFile:
    lineTokenized = line.split() #Splits the current line into tokens
    currentSpeaker = lineTokenized[0] #Holds the speaker of the current line
    currentSpeakerWords = [lineTokenized[x] for x in range(1,len(lineTokenized)) if lineTokenized[x] not in string.punctuation] #Holds a list of the words [DOES NOT INCLUDE PUNCTUATION] said by the speaker of the current line (in order)

    classLikelihoodList = {} #Holds likelihood each speaker said the current line
    for key, value in totalBigramsPerSpeaker.items(): #Go through each speaker from the training set
        tempSpeaker = key
        tempProbability = 0.0 #Initialize the probability of the current speaker having said the current line
        prior = math.log( docsPerClass[tempSpeaker] / float(totalNumberOfDocs) ) #Likelihood of the class(speaker) [a.k.a the prior]
        tempProbability += prior #Add the prior to the probability

        for i in range(len(currentSpeakerWords)): #Go through each word that the current speaker is saying
            if(i != 0):
                tempProbability += math.log( float(bigramCountsPerSpeaker[tempSpeaker][ (currentSpeakerWords[i-1], currentSpeakerWords[i]) ]) / totalBigramsPerSpeaker[tempSpeaker] ) #Add probability of word given class (speaker)
            elif(i == 0):
                tempProbability += math.log( float(bigramCountsPerSpeaker[tempSpeaker][ ('<s>', currentSpeakerWords[i]) ]) / totalBigramsPerSpeaker[tempSpeaker] ) #Add probability of word given class (speaker)
            if(i == len(currentSpeakerWords) - 1):
                tempProbability += math.log( float(bigramCountsPerSpeaker[tempSpeaker][ (currentSpeakerWords[i], '</s>') ]) / totalBigramsPerSpeaker[tempSpeaker] ) #Add probability of word given class (speaker)

        if(tempSpeaker not in classLikelihoodList):
            classLikelihoodList[tempSpeaker] = tempProbability #Initializes probability of each speaker to 1

    predictedSpeaker = max(classLikelihoodList, key = classLikelihoodList.get) #Sets prediction equal to the speaker with the highest likelihood for the line

    if(predictedSpeaker == currentSpeaker): #Append a 1 for a correct prediction
        devAccuracyList.append(1)
    else:                                   #Append a 0 for an incorrect prediction
        devAccuracyList.append(0)

print("Accuracy of bigram model on dev set: ", devAccuracyList.count(1) / len(devAccuracyList)) #Prints Accuracy of Model on Dev set (Number of correct predictions / Total Number of Predictions)
###END OF DEV SET RESULTS

################### TEST SET RESULTS ##########################################################################################################################################################
testAccuracyList = [] #Will hold 1 for correct prediction of a line's class(speaker), and 0 for incorrect prediction of a line's class(speaker)

testFile = open('test', 'r') #Test set
for line in testFile:
    lineTokenized = line.split() #Splits the current line into tokens
    currentSpeaker = lineTokenized[0] #Holds the speaker of the current line
    currentSpeakerWords = [lineTokenized[x] for x in range(1,len(lineTokenized)) if lineTokenized[x] not in string.punctuation] #Holds a list of the words [DOES NOT INCLUDE PUNCTUATION] said by the speaker of the current line (in order)

    classLikelihoodList = {} #Holds likelihood each speaker said the current line
    for key, value in totalBigramsPerSpeaker.items(): #Go through each speaker from the training set
        tempSpeaker = key
        tempProbability = 0.0 #Initialize the probability of the current speaker having said the current line
        prior = math.log( docsPerClass[tempSpeaker] / float(totalNumberOfDocs) ) #Likelihood of the class(speaker) [a.k.a the prior]
        tempProbability += prior #Add the prior to the probability

        for i in range(len(currentSpeakerWords)): #Go through each word that the current speaker is saying
            if(i != 0):
                tempProbability += math.log( float(bigramCountsPerSpeaker[tempSpeaker][ (currentSpeakerWords[i-1], currentSpeakerWords[i]) ]) / totalBigramsPerSpeaker[tempSpeaker] ) #Add probability of word given class (speaker)
            elif(i == 0):
                tempProbability += math.log( float(bigramCountsPerSpeaker[tempSpeaker][ ('<s>', currentSpeakerWords[i]) ]) / totalBigramsPerSpeaker[tempSpeaker] ) #Add probability of word given class (speaker)
            if(i == len(currentSpeakerWords) - 1):
                tempProbability += math.log( float(bigramCountsPerSpeaker[tempSpeaker][ (currentSpeakerWords[i], '</s>') ]) / totalBigramsPerSpeaker[tempSpeaker] ) #Add probability of word given class (speaker)

        if(tempSpeaker not in classLikelihoodList):
            classLikelihoodList[tempSpeaker] = tempProbability #Initializes probability of each speaker to 1

    predictedSpeaker = max(classLikelihoodList, key = classLikelihoodList.get) #Sets prediction equal to the speaker with the highest likelihood for the line

    if(predictedSpeaker == currentSpeaker): #Append a 1 for a correct prediction
        testAccuracyList.append(1)
    else:                                   #Append a 0 for an incorrect prediction
        testAccuracyList.append(0)

print("Accuracy of bigram model on test set: ", testAccuracyList.count(1) / len(testAccuracyList)) #Prints Accuracy of Model on Test set (Number of correct predictions / Total Number of Predictions)
###END OF TEST SET RESULTS










############################################# (Part B) STEMMERS (SNOWBALL) TESTING #################################################################################
totalNumberOfDocs = 0 #HOLDS THE TOTAL NUMBER OF LINES
totalUniqueStems = 0 #HOLDS NUMBER OF UNIQUE STEMS STATED BY EVERYONE ALTOGETHER
listOfUniqueStems = {} #Holds the unique stems of ALL the speakers COMBINED
docsPerClass = {} #Holds how many lines each person has spoken
stemCountsPerSpeaker = {} #Holds how many times each speaker has said a stem from all of the possible stems (Add-One smoothing is applied)
totalStemsPerSpeaker = {} #Holds counts for total stems per speaker (with unique vocabulary of ALL speakers added in)
snowball = SnowballStemmer('english')

file = open('train', 'r') #Opens the training file
for line in file:
    lineTokenized = line.split() #Splits the current line into tokens
    currentSpeaker = lineTokenized[0] #Holds the speaker of the current line
    currentSpeakerStems = [snowball.stem(lineTokenized[x]) for x in range(1,len(lineTokenized)) if lineTokenized[x] not in string.punctuation] #Holds a list of the stems of words [DOES NOT INCLUDE PUNCTUATION] said by the speaker of the current line (in order)

    if(currentSpeaker not in docsPerClass): #If the current speaker is not in docsPerClass
        docsPerClass[currentSpeaker] = 1 #Initialize
    else:
        docsPerClass[currentSpeaker] += 1 #Add 1 to the number of documents for the current speaker
    totalNumberOfDocs += 1 #Increases the number of documents regardless

    if (currentSpeaker not in stemCountsPerSpeaker):  # If the current speaker is not in stemCountsPerSpeaker
        stemCountsPerSpeaker[currentSpeaker] = collections.defaultdict(lambda: 1)  # Add-One smoothing to account for stems each speaker has not said
        for i in range(len(currentSpeakerStems)):
            token = currentSpeakerStems[i]  # Current stem in the list
            stemCountsPerSpeaker[currentSpeaker][token] = stemCountsPerSpeaker[currentSpeaker][token] + 1  # The count for the current stem in the list is increased by 1 for the current speaker
    else:
        for i in range(len(currentSpeakerStems)):
            token = currentSpeakerStems[i]  # Current stem in the list
            stemCountsPerSpeaker[currentSpeaker][token] = stemCountsPerSpeaker[currentSpeaker][token] + 1  # The count for the current stem in the list is increased by 1 for the current speaker

for key, value in stemCountsPerSpeaker.items():
    speaker = key
    totalStemsPerSpeaker[speaker] = 0 #No need to check if speaker is already in list because stemCountsPerSpeaker has each speaker listed once
    for key, value in stemCountsPerSpeaker[speaker].items(): #Go through all the stems of each speaker
        currentStem = key
        totalStemsPerSpeaker[speaker] += value #Increase that speakers total stem count accordingly

        if(currentStem not in listOfUniqueStems): #If a stem said by a given speaker is not in the list of unique stems
            listOfUniqueStems[currentStem] = 1

for key, value in listOfUniqueStems.items(): #Counting up the total number of unique stems used by ALL the speakers combined
    totalUniqueStems += 1

for key, value in totalStemsPerSpeaker.items():
    totalStemsPerSpeaker[key] += totalUniqueStems #This is used for Add-One smoothing, in which the unique vocabulary of ALL the speakers is added

################### DEV SET RESULTS ##########################################################################################################################################################
devAccuracyList = [] #Will hold 1 for correct prediction of a line's class(speaker), and 0 for incorrect prediction of a line's class(speaker)

devFile = open('dev', 'r') #Dev set
for line in devFile:
    lineTokenized = line.split() #Splits the current line into tokens
    currentSpeaker = lineTokenized[0] #Holds the speaker of the current line
    currentSpeakerStems = [snowball.stem(lineTokenized[x]) for x in range(1,len(lineTokenized)) if lineTokenized[x] not in string.punctuation] #Holds a list of the stems of words [DOES NOT INCLUDE PUNCTUATION] said by the speaker of the current line (in order)

    classLikelihoodList = {} #Holds likelihood each speaker said the current line
    for key, value in totalStemsPerSpeaker.items(): #Go through each speaker from the training set
        tempSpeaker = key
        tempProbability = 0.0 #Initialize the probability of the current speaker having said the current line
        prior = math.log( docsPerClass[tempSpeaker] / float(totalNumberOfDocs) ) #Likelihood of the class(speaker) [a.k.a the prior]
        tempProbability += prior #Add the prior to the probability

        for i in range(len(currentSpeakerStems)): #Go through each stem that the current speaker is saying
            tempProbability += math.log( float(stemCountsPerSpeaker[tempSpeaker][currentSpeakerStems[i]]) / totalStemsPerSpeaker[tempSpeaker] ) #Add probability of word given class (speaker)

        if(tempSpeaker not in classLikelihoodList):
            classLikelihoodList[tempSpeaker] = tempProbability #Initializes probability of each speaker to 1

    maxProbability = max(classLikelihoodList.values())
    predictedSpeaker = [key for key in classLikelihoodList if classLikelihoodList[key] == maxProbability] #Holds speakers with highest likelihood (more than 1 if tie)
    #predictedSpeaker = max(classLikelihoodList, key = classLikelihoodList.get) #Sets prediction equal to the speaker with the highest likelihood for the line

    if(currentSpeaker in predictedSpeaker): #Append a 1 for a correct prediction
        devAccuracyList.append(1)
    else:                                   #Append a 0 for an incorrect prediction
        devAccuracyList.append(0)

print("Accuracy of snowball stemmer model on dev set: ", devAccuracyList.count(1) / len(devAccuracyList)) #Prints Accuracy of Model on Dev set (Number of correct predictions / Total Number of Predictions)
###END OF DEV SET RESULTS

################### TEST SET RESULTS ##########################################################################################################################################################
testAccuracyList = [] #Will hold 1 for correct prediction of a line's class(speaker), and 0 for incorrect prediction of a line's class(speaker)

testFile = open('test', 'r') #Test set
for line in testFile:
    lineTokenized = line.split() #Splits the current line into tokens
    currentSpeaker = lineTokenized[0] #Holds the speaker of the current line
    currentSpeakerStems = [snowball.stem(lineTokenized[x]) for x in range(1,len(lineTokenized)) if lineTokenized[x] not in string.punctuation] #Holds a list of the stems of words [DOES NOT INCLUDE PUNCTUATION] said by the speaker of the current line (in order)

    classLikelihoodList = {} #Holds likelihood each speaker said the current line
    for key, value in totalStemsPerSpeaker.items(): #Go through each speaker from the training set
        tempSpeaker = key
        tempProbability = 0.0 #Initialize the probability of the current speaker having said the current line
        prior = math.log( docsPerClass[tempSpeaker] / float(totalNumberOfDocs) ) #Likelihood of the class(speaker) [a.k.a the prior]
        tempProbability += prior #Add the prior to the probability

        for i in range(len(currentSpeakerStems)): #Go through each stem that the current speaker is saying
            tempProbability += math.log( float(stemCountsPerSpeaker[tempSpeaker][currentSpeakerStems[i]]) / totalStemsPerSpeaker[tempSpeaker] ) #Add probability of word given class (speaker)

        if(tempSpeaker not in classLikelihoodList):
            classLikelihoodList[tempSpeaker] = tempProbability #Initializes probability of each speaker to 1

    maxProbability = max(classLikelihoodList.values())
    predictedSpeaker = [key for key in classLikelihoodList if classLikelihoodList[key] == maxProbability] #Holds speakers with highest likelihood (more than 1 if tie)
    #predictedSpeaker = max(classLikelihoodList, key = classLikelihoodList.get) #Sets prediction equal to the speaker with the highest likelihood for the line

    if(currentSpeaker in predictedSpeaker): #Append a 1 for a correct prediction
        testAccuracyList.append(1)
    else:                                   #Append a 0 for an incorrect prediction
        testAccuracyList.append(0)

print("Accuracy of snowball stemmer model on test set: ", testAccuracyList.count(1) / len(testAccuracyList)) #Prints Accuracy of Model on Test set (Number of correct predictions / Total Number of Predictions)
###END OF TEST SET RESULTS