import os
from watson_developer_cloud import VisualRecognitionV3
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras import Sequential
import cv2
import numpy as np
import shutil

stopwords = stopwords.words('english')
my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
my_new_model.layers[0].trainable = False


def ibm(path):
    # path = 'Resources/char.jpg'
    dic = []

    visual_recognition = VisualRecognitionV3(version='2016-05-20',
                                             iam_apikey='_BWRFUuaOZeqti8Vhj85iRSb2xhPew6gfTRdU4KTSllx')  # authenticate
    classes = visual_recognition.classify(images_file=open(path, 'rb'), threshold='0.85',
                                          classifier_ids='default').get_result()
    try:
        c = len(classes['images'][0]['classifiers'][0]['classes'])
        for i in range(c):
            dic.append(classes['images'][0]['classifiers'][0]['classes'][i]['class'])
        return dic
    except:
        pass


def clean(names):
    clean = []
    for i in names:
        x = i.replace("_", " ")
        x = x.replace("(", "")
        x = x.replace(")", "")
        x = x.lower()
        clean.append(x)
    return clean


def cs(v1, v2):
    v1 = v1.reshape(1, -1)
    v2 = v2.reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]


def folder_sim(test):
    final = {}
    names = os.listdir("output")
    dic = ibm(test)
    dicf = []
    snow_stemmer = SnowballStemmer(language='english')
    for i in dic:
        x = snow_stemmer.stem(i)
        dicf.append(x)
    names.append('_'.join(dicf))
    clea = clean(names)
    vec = CountVectorizer().fit_transform(clea)
    v = vec.toarray()

    for i in range(10):
        final[names[i]] = cs(v[i], v[10])

    final = dict(sorted(final.items(), key=lambda item: item[1], reverse=True))

    x = list(final.keys())
    return x[:3]


def extract_vector(path):
    resnet_feature_list = []
    ifold = []
    for im in os.listdir(path):
        ifold.append(im)
        im = cv2.imread(path + "//" + im)
        im = cv2.resize(im, (224, 224))
        img = preprocess_input(np.expand_dims(im.copy(), axis=0))
        resnet_feature = my_new_model.predict(img)
        resnet_feature_np = np.array(resnet_feature)
        resnet_feature_list.append(resnet_feature_np.flatten())

    return ifold, np.array(resnet_feature_list)


top = folder_sim("input//test.jpg")
ifold1, f1 = extract_vector("input")
score = {}
for i in top:
    ifold, f = extract_vector("output//" + i)
    for img, j in zip(ifold, f):
        sim = cs(f1, j)
        score["output//" + i + "//" + img] = sim

score = dict(sorted(score.items(), key=lambda item: item[1], reverse=True))

c = 0
for i, j in score.items():
    if c == 10:
        break
    else:
        print(i, "==>", j)
        shutil.copyfile(i, "result//"+str(j)+".jpg")
        c = c + 1
