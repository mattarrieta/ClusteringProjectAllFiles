from CleanText import getText

text_l, labels = getText(r"C:\Users\Matthew Arrieta\Desktop\Project3Testing\TestFiles\KeywordFilesTight", RemoveNums = True, lem=True, extendStopWords= False)
print(labels)

