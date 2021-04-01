import os
from watson_developer_cloud import VisualRecognitionV3
from nltk.stem.snowball import SnowballStemmer


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


def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list


def rename(num_cluster):
    snow_stemmer = SnowballStemmer(language='english')

    for i in range(num_cluster):
        f = []
        for j in os.listdir("output//" + str(i)):
            x = ibm("output//" + str(i) + "//" + j)
            f.append(x)
        ff = flatten(f)
        ff = list(set(ff))
        print(ff)
        fff = []
        for k in ff:
            x = snow_stemmer.stem(k)
            fff.append(x)
        print(fff)
        fname = '_'.join(fff)
        print(fname)
        os.rename("output//" + str(i), "output//" + fname)


# rename(10) ####Rename all the cluster folders