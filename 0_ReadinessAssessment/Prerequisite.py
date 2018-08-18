from collections import Counter

def make_list_words(sentence):
    charcount = 0
    listwords = []
    for i in range(len(sentence)):
        if sentence[i] == ' ':
            listwords += [sentence[charcount:i]]
            charcount = i+1
    listwords += [sentence[charcount:len(sentence)]]
    return listwords

print make_list_words("my list of my words")
print "my list of my words".split()

def count_words(s, n):
    sortedlistwords = sorted(s.split())
    nwords = len(sortedlistwords)
    wordcount = 1
    listwordsandcounts = []
    if nwords==1:
        listwordsandcounts=[(sortedlistwords[0],1)]
    else:
        for i in range(1, nwords):
            if sortedlistwords[i] == sortedlistwords[i-1]:
                wordcount += 1
            else:
                listwordsandcounts += [(sortedlistwords[i-1], wordcount)]
                wordcount = 1
        listwordsandcounts += [(sortedlistwords[nwords-1], wordcount)]
    listwordsandcounts.sort(key=lambda tup: (-tup[1], tup[0]))

    top_n = listwordsandcounts[0:n]

    return top_n

print count_words("my list of my word of the words", 7)
print sorted(Counter("my list of my word of the words".split()).items(), key=lambda tup: (-tup[1], tup[0]))
