# WORD ADJACENCY NETWORKs -- A Python Script
# (This version does not stop the 5-word window at speech-breaks.)
#
# In the same folder as this program there should exist three
# ASCII-format files:
#
# 1) "text1.txt"
# 2) "text2.txt"
# 3) "function_words.
#
# The files text1 and text2 will typically be literary or historical
# documents and no special encoding is presumed. Thus the opening
# of Shakespeare's play 'The Tempest' could be encoded thus, as it appears
# in a simple original-spelling transcription from 1623 First Folio:
# 
#   Actus primus, Scena prima.
#   A tempestuous noise of Thunder and Lightning heard: Enter
#   a Ship-master, and a Boteswaine.
# 
#   Master.
#   Bote-swaine.
#
#   Botes.
#   Heere Master: What cheere?
#
#   Mast.
#   Good: Speake to th' Mariners: fall too't, yarely, or we
#   run our selues a ground, bestirre, bestirre. Exit.
#   ...
#
# The file "function_words.txt" should be a simple list of words
# separated by a space, as in:
#
#   and the in on with but for
#
# The program will output to the console a running commentary
# on the comparisons it makes, leading to a final statement of
# the relative entropy between text1 and text2. For files of
# anything more than a trivial size (dozens of words) this
# running commentary can run to thousands of outputted lines
# and it is recommended to redirect the console output to a
# file to be consulted after the program is finished. On the
# Microsoft Windows command line this is achieved using the
# redirection operator ">", as in "python WAN.py > out.txt".
# 
# The three file names used can be changed by editing their
# occurrences in lines 253, 254, and 258 below. The size of
# 5-word window can be changed by editing the variable assignment
# in line 255 below.
###################################################################

from math import log

def cleanUp(anytext):           # This does the removal of punctuation and lowercasing of the list 'anytext'
    print("\nEntering cleanUp()")
    punctuation_to_remove = [',', '.', '!', '?', '"', ':', ';', ')', '(', '[', ']']
    print("Punctuation that will be removed: "+str(punctuation_to_remove))
    for mark in (punctuation_to_remove):
        anytext = anytext.replace(mark, "")
    print("Text after punctuation and paragraphs processed:\n\n"+anytext)
    anytext = anytext.lower().split()
    print("Text after lowercased and made into list of strings:")
    print(anytext)
    print("length of text is "+str(len(anytext)))
    print("End of cleanUp()")
    return anytext

def showMatrix(anyMatrix):      # This prints our square matrix in two dimensions so it's easier to read
    if anyMatrix == []:
        print("Empty")
        return
    else:
        for i in range (0,len(anyMatrix[0])):
            print("            "+str(anyMatrix[i]))

def countFWs(anyText, anyFwords, anyWindowWidth, anyInputFileName): # This generates the matrix of
                                                                    #   weighted scores (pre-normalization
    print("\nEntering countFWs()\n")                                #   and calculation of limit probabilities)
                                                                    # We pass the input file name merely so we can
                                                                    #   display it in the console log output.
                                                                    # This function returns the weighted scores as
                                                                    #   a square matrix and also the raw counts
                                                                    #   of function word hits as a 1D array to be
                                                                    #   used later in limitProbabilities()
    
    fwordsScores = []                                               # Create an empty square matrix
    for z in range(0, len(anyFwords)):                              #   to hold the weights of the MC edges
        fwordsScores.append([0] * len(anyFwords))                   #   from each node to each other
    print("This is the matrix (fwordScores) holding the weighted counts for each function word:")
    showMatrix(fwordsScores)
    
    fwordsRawCounts = []                    # As well as the weighted scores we need integer counts of hits
    for z in range(0, len(anyFwords)):      #   which we will use in the limit-probabilities calculation
        fwordsRawCounts.append(0)           # We create it as an empty 1-D array (= Python list) and then
                                            #   append values of '0' as many times as there are function words
    print("\nWe use a 1-dimension array (fwordsRawCounts) to hold raw integer counts of how often each function word found in text:")
    for z in range(0, len(anyFwords)):
        print("The starting count for function word '"+anyFwords[z]+"' is: "+str(fwordsRawCounts[z]))
    

    for i in range(0, len(anyText)-1):      # iterate thru text (don't check last word: no room to open window)
        print("\nIn "+anyInputFileName+" word number "+str(i)+" is '"+anyText[i]+"'")
        for j in range(0, len(anyFwords)):  # for this word in text, iterate thru the list of function words
            print("  Function word list number "+str(j)+" is '"+anyFwords[j]+"'")
            if anyText[i] == anyFwords[j]:  # we got a match, so open the window
                print("    Hit: source text word '"+anyText[i]+"' = function-word list word '"+anyFwords[j]+"'.")
                fwordsRawCounts[j] = fwordsRawCounts[j]+1
                print("    That means we have seen '"+anyFwords[j]+"' "+str(fwordsRawCounts[j])+" times.")
                print("    Let's open a window.")
                if len(anyText)-i <= anyWindowWidth:
                    thisWindowWidth = len(anyText)-1-i
                else:
                    thisWindowWidth = anyWindowWidth
                print("    Using window of width "+str(thisWindowWidth))
                for k in range(0, thisWindowWidth):
                    print("        Looking in window at word number "+str(k)+" which is '"+anyText[i+1+k]+"'")
#                   if anyText[i+1+k] == "xxxxx":
#                       print("          That's an end-of-speech marker, so terminating the window trawl.")
#                       break
                    for l in range(0, len(anyFwords)):
                        print("          Comparing it to function word '"+anyFwords[l]+"'")
                        if anyText[i+1+k] == anyFwords[l]:
                            print("            Match!, so row for '"+anyFwords[j]+"' gets raised in its column for '"+anyFwords[l]+"' by 0.75 to the power of "+str(k))
                            print("            That is, row "+str(j)+" and column "+str(l)+" is raised by "+str(0.75**k)+" giving us:")
                            fwordsScores[j][l] = fwordsScores[j][l] + (0.75 ** k)
                            showMatrix(fwordsScores)
                            print("            Continuing window trawl.")
                            break           # If we match this word in the window with a function word, no point
                                            #   looking at further function words to match it with
                break               # Having matched current word in text (not in the window, in the text)
                                    #   with a function word and looked ahead, there's no point continuing
                                    #   to compare that current word with the other function words
    print("\nOutput of countFWs():")
    showMatrix(fwordsScores)
    print("fwordsRawCounts: "+str(fwordsRawCounts))
    print("\n\nEnd of countFWs()\n")
    return (fwordsScores, fwordsRawCounts)

def normalize(anyMatrix):                               # divide each cell in a square matrix by the sum of the row 
    print("Entering normalize()")                       #   it's in (unless that sum is 0)
    for i in range(0,len(anyMatrix[0])):                # iterate thru rows
        rowSum=0
        for j in range(0,len(anyMatrix[0])):                # iterate thru columns in this row
            rowSum = rowSum + anyMatrix[i][j]               # build rowSum
        if rowSum != 0:                                     # once rowSum calculated, unless it's zero
            for j in range(0,len(anyMatrix[0])):            #   iterate thru same row again, dividing
                anyMatrix[i][j] = anyMatrix[i][j]/rowSum    #   each cell by its row's sum
    print("\nOutput of normalize:")
    showMatrix(anyMatrix)
    print("End of normalize()")
    return anyMatrix

# def eliminateSinks(anyMatrix):          # If any row in a square matrix is all zeroes, that indicates a
#                                         #   function word not followed (with our window) by any other function
#                                         #   words. The limit probability of such a 'sink' would be 1 and all
#                                         #   other limit probabilities would be 0, since there is no way out of
#                                         #   this state. To fix this we set all the cells in that row to 1 divided
#                                         #   by the number width ( = the height) of the matrix.
#     print("\nEntering eliminateSinks()")
#     print("input anyMatrix is")
#     showMatrix(anyMatrix)

#     dummyMatrix=[]                                      # Because of the weird way Python treats "=" we need to take
#     for i in range(0, len(anyMatrix[0])):               #   a dummy copy of the input matrix and work only on that
#         dummyMatrix.append([0] * len(anyMatrix[0]))     # So first we create the copy and populate it with zeroes
    
#     for i in range(0,len(anyMatrix[0])):                            # iterate thru rows
#         rowSum=0
#         for j in range(0,len(anyMatrix[0])):                        # iterate thru columns in this row
#             rowSum = rowSum + anyMatrix[i][j]                       # build rowSum
#         if rowSum == 0:                                             # once rowSum calculated, if it's zero
#             for j in range(0,len(anyMatrix[0])):                    #   iterate thru same row again, setting each
#                 dummyMatrix[i][j] = 1 / len(anyMatrix[0])           #   cell in dummyMatrix to 1/matrix-width
#         else:
#             for j in range(0,len(anyMatrix[0])):                    # or else just copy the anyMatrix value
#                 dummyMatrix[i][j] = anyMatrix[i][j]                 #   over to the new dummyMatrix

        
#     print("\nOutput of eliminateSinks:")
#     showMatrix(dummyMatrix)
    
#     print("anyMatrix is still")
#     showMatrix(anyMatrix)
    
#     print("End of eliminateSinks()")
#     return dummyMatrix                              
    
############################### End of function definitions ##################

windowWidth = 5

inputFile_shorter = "test_wan/test_short.txt"
outputFile_shorter = "test_wan/wan_test_short.txt"
functionWordsFile_shorter = "test_wan/function_words_short.txt"

inputFile_longer = "test_wan/test_longer.txt"
outputFile_longer = "test_wan/wan_test_longer.txt"
functionWordsFile_longer = "test_wan/function_words.txt"

def construct_wan(inputFile, windowWidth, outputFile, functionWordsFile):
    print("\n\nBEGINNING ...")

    with open(functionWordsFile, 'r') as handle:
        fwords = sorted(handle.read().split())
    print("List of function words: "+str(fwords))

    with open(inputFile, "r") as handle:
        text=handle.read()     # we no longer replace whitespace between paragraphs with 'xxxxx'
        
    print("\n\nDoing "+inputFile)
    (textScores, textCounts) = countFWs(cleanUp(text),fwords,windowWidth, inputFile) # We pass countFWs() the input
                                                                                        #   filename so it can show it
    WAN = normalize(textScores) 

    with open(outputFile, "w") as f:
        for i in range(0, len(fwords)):
            for j in range(0, len(fwords)):
                f.write(str(WAN[i][j]) + " ")
            f.write("\n")

    return WAN

WAN_shorter = construct_wan(inputFile_shorter, windowWidth, outputFile_shorter, functionWordsFile_shorter)
WAN_longer = construct_wan(inputFile_longer, windowWidth, outputFile_longer, functionWordsFile_longer)