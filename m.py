from transformers import pipeline
classifier = pipeline("sentiment-analysis") # model is passed
classifier2 = pipeline("text-classification")
res = classifier("This is the beginning Alhamdulillah")
res2neg = classifier("This isn't the right way")
resNew = classifier("Naujubillah")
resbook1 = classifier("I guess you are not ok")
resbook2 = classifier("I guess you are ok")
# print(res2neg)
# print(classifier2)
#print(resNew)
print(resbook1)
print(resbook2)