import numpy as np 
import torch 
from math import log


####
# TO-DO => Misc. for visualization, metrics
####

##############
# Code For WAN creation courtesy of course suggested materials and can
# be found at http://gabrielegan.com/WAN/index.html
##############

# Added for the training pipeline: the original script prints every comparison it makes,
# which makes one essay take ~0.3 s instead of ~0.02 s. Every print below now goes through
# vprint(), which only prints when VERBOSE is True. The default keeps the old behaviour;
# the training pipeline switches it off (vis.utils.VERBOSE = False).
VERBOSE = True

def vprint(*args):
    if VERBOSE:
        print(*args)

def cleanUp(anytext):           # This does the removal of punctuation and lowercasing of the list 'anytext'
    vprint("\nEntering cleanUp()")
    punctuation_to_remove = [',', '.', '!', '?', '"', ':', ';', ')', '(', '[', ']']
    vprint("Punctuation that will be removed: "+str(punctuation_to_remove))
    for mark in (punctuation_to_remove):
        anytext = anytext.replace(mark, "")
    vprint("Text after punctuation and paragraphs processed:\n\n"+anytext)
    anytext = anytext.lower().split()
    vprint("Text after lowercased and made into list of strings:")
    vprint(anytext)
    vprint("length of text is "+str(len(anytext)))
    vprint("End of cleanUp()")
    return anytext

def showMatrix(anyMatrix):      # This prints our square matrix in two dimensions so it's easier to read
    if not VERBOSE:             # added: skip building the 117-row printout when not printing
        return
    if anyMatrix == []:
        vprint("Empty")
        return
    else:
        for i in range (0,len(anyMatrix[0])):
            vprint("            "+str(anyMatrix[i]))

def countFWs(anyText, anyFwords, anyWindowWidth, anyInputFileName): # This generates the matrix of
                                                                    #   weighted scores (pre-normalization
    vprint("\nEntering countFWs()\n")                                #   and calculation of limit probabilities)
                                                                    # We pass the input file name merely so we can
                                                                    #   display it in the console log output.
                                                                    # This function returns the weighted scores as
                                                                    #   a square matrix and also the raw counts
                                                                    #   of function word hits as a 1D array to be
                                                                    #   used later in limitProbabilities()
    
    fwordsScores = []                                               # Create an empty square matrix
    for z in range(0, len(anyFwords)):                              #   to hold the weights of the MC edges
        fwordsScores.append([0] * len(anyFwords))                   #   from each node to each other
    vprint("This is the matrix (fwordScores) holding the weighted counts for each function word:")
    showMatrix(fwordsScores)
    
    fwordsRawCounts = []                    # As well as the weighted scores we need integer counts of hits
    for z in range(0, len(anyFwords)):      #   which we will use in the limit-probabilities calculation
        fwordsRawCounts.append(0)           # We create it as an empty 1-D array (= Python list) and then
                                            #   append values of '0' as many times as there are function words
    vprint("\nWe use a 1-dimension array (fwordsRawCounts) to hold raw integer counts of how often each function word found in text:")
    for z in range(0, len(anyFwords)):
        vprint("The starting count for function word '"+anyFwords[z]+"' is: "+str(fwordsRawCounts[z]))
    

    for i in range(0, len(anyText)-1):      # iterate thru text (don't check last word: no room to open window)
        vprint("\nIn "+anyInputFileName+" word number "+str(i)+" is '"+anyText[i]+"'")
        for j in range(0, len(anyFwords)):  # for this word in text, iterate thru the list of function words
            vprint("  Function word list number "+str(j)+" is '"+anyFwords[j]+"'")
            if anyText[i] == anyFwords[j]:  # we got a match, so open the window
                vprint("    Hit: source text word '"+anyText[i]+"' = function-word list word '"+anyFwords[j]+"'.")
                fwordsRawCounts[j] = fwordsRawCounts[j]+1
                vprint("    That means we have seen '"+anyFwords[j]+"' "+str(fwordsRawCounts[j])+" times.")
                vprint("    Let's open a window.")
                if len(anyText)-i <= anyWindowWidth:
                    thisWindowWidth = len(anyText)-1-i
                else:
                    thisWindowWidth = anyWindowWidth
                vprint("    Using window of width "+str(thisWindowWidth))
                for k in range(0, thisWindowWidth):
                    vprint("        Looking in window at word number "+str(k)+" which is '"+anyText[i+1+k]+"'")
#                   if anyText[i+1+k] == "xxxxx":
#                       vprint("          That's an end-of-speech marker, so terminating the window trawl.")
#                       break
                    for l in range(0, len(anyFwords)):
                        vprint("          Comparing it to function word '"+anyFwords[l]+"'")
                        if anyText[i+1+k] == anyFwords[l]:
                            vprint("            Match!, so row for '"+anyFwords[j]+"' gets raised in its column for '"+anyFwords[l]+"' by 0.75 to the power of "+str(k))
                            vprint("            That is, row "+str(j)+" and column "+str(l)+" is raised by "+str(0.75**k)+" giving us:")
                            fwordsScores[j][l] = fwordsScores[j][l] + (0.75 ** k)
                            showMatrix(fwordsScores)
                            vprint("            Continuing window trawl.")
                            break           # If we match this word in the window with a function word, no point
                                            #   looking at further function words to match it with
                break               # Having matched current word in text (not in the window, in the text)
                                    #   with a function word and looked ahead, there's no point continuing
                                    #   to compare that current word with the other function words
    vprint("\nOutput of countFWs():")
    showMatrix(fwordsScores)
    vprint("fwordsRawCounts: "+str(fwordsRawCounts))
    vprint("\n\nEnd of countFWs()\n")
    return (fwordsScores, fwordsRawCounts)

def normalize(anyMatrix):                               # divide each cell in a square matrix by the sum of the row 
    vprint("Entering normalize()")                       #   it's in (unless that sum is 0)
    for i in range(0,len(anyMatrix[0])):                # iterate thru rows
        rowSum=0
        for j in range(0,len(anyMatrix[0])):                # iterate thru columns in this row
            rowSum = rowSum + anyMatrix[i][j]               # build rowSum
        if rowSum != 0:                                     # once rowSum calculated, unless it's zero
            for j in range(0,len(anyMatrix[0])):            #   iterate thru same row again, dividing
                anyMatrix[i][j] = anyMatrix[i][j]/rowSum    #   each cell by its row's sum
    vprint("\nOutput of normalize:")
   # showMatrix(anyMatrix)
    vprint("End of normalize()")
    return anyMatrix

def construct_wan(inputFile, windowWidth, outputFile, functionWordsFile):
    vprint("\n\nBEGINNING ...")

    with open(functionWordsFile, 'r') as handle:
        fwords = sorted(handle.read().split())
    vprint("List of function words: "+str(fwords))

    with open(inputFile, "r") as handle:
        text=handle.read()     # we no longer replace whitespace between paragraphs with 'xxxxx'
        
    vprint("\n\nDoing "+inputFile)
    (textScores, textCounts) = countFWs(cleanUp(text),fwords,windowWidth, inputFile) # We pass countFWs() the input
                                                                                        #   filename so it can show it
    WAN = normalize(textScores) 

    with open(outputFile, "w") as f:
        for i in range(0, len(fwords)):
            for j in range(0, len(fwords)):
                f.write(str(WAN[i][j]) + " ")
            f.write("\n")

    return WAN

def construct_wan_from_text(inputText, windowWidth, functionWordsFile, textLabel="input string",
                            return_counts=False):
    vprint("\n\nBEGINNING ...")

    with open(functionWordsFile, 'r') as handle:
        fwords = sorted(handle.read().split())
    vprint("List of function words: "+str(fwords))

    vprint("\n\nDoing "+textLabel)
    words = cleanUp(inputText)
    (textScores, textCounts) = countFWs(words,fwords,windowWidth, textLabel) # We pass countFWs() a label
                                                                              #   so it can show it
    WAN = normalize(textScores)

    if return_counts:   # added: the GNN uses each word's count (node feature) and the text length
        return WAN, textCounts, len(words)
    return WAN

if  __name__ == "__main__":
    windowWidth = 5

    inputFile_shorter = "test_wan/test_short.txt"
    outputFile_shorter = "test_wan/wan_test_short.txt"
    functionWordsFile_shorter = "test_wan/function_words_short.txt"

    inputFile_longer = "test_wan/test_longer.txt"
    outputFile_longer = "test_wan/wan_test_longer.txt"
    functionWordsFile_longer = "test_wan/function_words.txt"


    vprint("Testing WAN construction ...")

    WAN_shorter = construct_wan(inputFile_shorter, windowWidth, outputFile_shorter, functionWordsFile_shorter)
    WAN_longer = construct_wan(inputFile_longer, windowWidth, outputFile_longer, functionWordsFile_longer)