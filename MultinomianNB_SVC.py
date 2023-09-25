import codecs
import zipfile
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, confusion_matrix

option = input("Multinomial Naive Bayes = 1; Mașini cu vectori suport = 2. Introduceti optiunea:")
def writeResults(testIDX, labels):
    with open("kaggle.csv", "w") as output:
        output.write("id,label\n")
        for p in zip(testIDX, labels):
            output.write(str(p[0])+','+str(p[1])+'\n')


def readSamples(zipFile, textFile):
    data = []
    with zipfile.ZipFile(zipFile) as thezip:
        with thezip.open(textFile, mode='r') as thefile:
                for line in codecs.iterdecode(thefile, 'utf8'):
                    idAndSentence = line.split("\t")
                    id, sentence = int(idAndSentence[0].strip()), str(idAndSentence[1].strip())
                    dataLine = {'id' : id, 'sentence': sentence }
                    data.append(dataLine)
    return pd.DataFrame(data)

def readLabels(filePath):
    data = []
    with open(filePath, mode='r',encoding="UTF-8") as thefile:
        for line in thefile.readlines():
                idAndLabel = line.split("\t")
                id, label = int(idAndLabel[0].strip()), int(idAndLabel[1].strip())
                dataLine = {'id': id, 'language': label}
                data.append(dataLine)
    return pd.DataFrame(data)


train_samples = readSamples(zipFile="train_samples.txt.zip", textFile="train_samples.txt")
train_labels = readLabels(filePath="train_labels.txt")
validation_samples = readSamples(zipFile="validation_samples.txt.zip",textFile="validation_samples.txt" )
validation_labels = readLabels(filePath="validation_labels.txt")
test_samples = readSamples(zipFile= "test_samples.txt.zip", textFile="test_samples.txt")

vectorizer = CountVectorizer(analyzer='char',ngram_range=(2,8), binary=True, encoding="UTF-8", lowercase=False, strip_accents = "unicode")

train_samples = train_samples.drop('id', axis=1)
train_labels = train_labels.drop('id', axis=1)
validation_labels = validation_labels.drop('id', axis=1)
validation_samples = validation_samples.drop('id', axis=1)
testIDX = test_samples.drop('sentence', axis=1)

train_samples = vectorizer.fit_transform(train_samples['sentence'])
validation_samples = vectorizer.transform(validation_samples['sentence'])
test_samples = vectorizer.transform(test_samples['sentence'])

if option == 2:
    classifier = svm.SVC(C=5, kernel= "linear", gamma="scale")
    classifier.fit(train_samples, train_labels) # antrenare
    predictions = classifier.predict(validation_samples)
    print(f1_score(validation_labels, predictions, average='macro'))
    labels = classifier.predict(test_samples)
else:
    classifier = MultinomialNB(alpha=0.2,fit_prior=True)
    classifier.fit(train_samples, train_labels)
    predictions = classifier.predict(validation_samples)
    print(f1_score(validation_labels, predictions, average='macro'))
    labels = classifier.predict(test_samples)

print(confusion_matrix(predictions, validation_labels))
writeResults(testIDX['id'], labels)

# 0.7651684359697467
# vectorizer = CountVectorizer(analyzer='char',ngram_range=(2,8), binary=True, encoding="UTF-8", lowercase=False, strip_accents = "unicode")
# classifier = MultinomialNB(alpha=0.2,fit_prior=True)

# 0.70335082868369
# classifier = MultinomialNB(alpha=0.1,fit_prior=True)
# vectorizer = CountVectorizer(analyzer='word',ngram_range=(1,10), binary=True, encoding="UTF-8")


# 0.759488074162559
# vectorizer = CountVectorizer(analyzer='char',ngram_range=(1,10), binary=True, encoding="UTF-8", strip_accents="unicode")
# classifier = MultinomialNB(alpha=0.3,fit_prior=True)

# 0.7644664447357888
# vectorizer = CountVectorizer(analyzer='char',ngram_range=(1,7), binary=True, encoding="UTF-8")
# classifier = MultinomialNB(alpha=0.1,fit_prior=True)

# 0.7474784963959363
# classifier = MultinomialNB(alpha=0.1,fit_prior=True)
# vectorizer = CountVectorizer(analyzer='char',ngram_range=(1,7), binary=True, encoding="UTF-8", strip_accents="ascii")

# 0.7650995741960854
# vectorizer = CountVectorizer(analyzer='char',ngram_range=(1,7), binary=True, encoding="UTF-8", strip_accents="unicode")
# classifier = MultinomialNB(alpha=0.1,fit_prior=True)

# # 0.7607096885228519
# classifier = MultinomialNB(alpha=0.5,fit_prior=True)
# vectorizer = CountVectorizer(analyzer='char',ngram_range=(1,8), binary=True, encoding="UTF-8")


# # 0.7640544257626006
# vectorizer = CountVectorizer(analyzer='char',ngram_range=(2,8), binary=True, encoding="UTF-8")
# classifier = MultinomialNB(alpha=0.2,fit_prior=True)

# 0.7253103356973932
# classifier = svm.SVC(C=1,kernel= "rbf", gamma=100)
# vec = CountVectorizer(analyzer="char", binary=True, encoding="UTF-8", ngram_range=(2,6))

# 0.6843664065346209
# vec = CountVectorizer(analyzer="char", binary=True, encoding="UTF-8", ngram_range=(1,2))
# classifier = svm.SVC(C=10,kernel= "poly", gamma=100)
