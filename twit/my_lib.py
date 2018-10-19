import re
import Stemmer


def splitstring(str):
    words = []
    str = str.lower()
    str = str.replace("ё", "е")
    stemmer = Stemmer.Stemmer('russian')
    # for i in re.split('[;,.,\n,\s,:,-,+,(,),=,/,«,»,\d,!,?,"]',str):
    # re.split("(?:(?:[^а-яА-Я]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)"
    for i in re.split("(?:[^а-я0-9]+)", str):
        if len(i) > 1 and len(i) <= 17:
            words.append(stemmer.stemWord(i))
            # words.append(i) # without stamming
    return words


def WordReplace(dataset, dict):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] in dict:
                dataset[i][j] = dict[dataset[i][j]]
            else:
                dataset[i][j] = 2  # unknown
    return dataset


# convert array to sentence. just for tests
def WordReturn(dataset, dict):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] in dict:
                print(dict[dataset[i][j]], end=' ')
            else:
                dataset[i][j] = 2  # unknown
        print()


def WordsDic(dataset):
    word = {"<PAD>": 0, "<START>": 1, "<UNK>": 2, "<UNUSED>": 3}
    index = 4
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] in word:
                None
            else:
                word.update({dataset[i][j]: index})
                index += 1
    return word
