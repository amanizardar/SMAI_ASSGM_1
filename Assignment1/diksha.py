import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag
from sklearn.model_selection import train_test_split
import string
import re
import math
from numpy import dot
from numpy.linalg import norm
from operator import itemgetter
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')




import pandas as pd
x = pd.read_csv('/content/emails.txt', sep="\t", header=None )
# print(x)
# df=pd.read_csv('emails.txt')
df_text=x[[1]]
df_text.head(100)





finalwords={}

def text_preprocessing(text):
    # convert text to lowercase
    
    # text = [char for char in text if char not in string.punctuation]
    # text = ''.join(text)
    # word tokenizing
    tokens = word_tokenize(text)
    # print(tokens)
    # removing noise: numbers, stopwords, and punctuation
    lang_stopwords = stopwords.words("english")
    tokens = [token for token in tokens if not token.isdigit() and \
                            not token in string.punctuation and \
                                token not in lang_stopwords ]

    finaltokens=[]
    # print(tokens)
    for each in tokens:
      if(each.isnumeric()==False ):
        temp=""
        for e in each:
          if(e not in string.punctuation and e not in ['0','1','2','3','4','5','6','7','8' ,'9']):
            temp+=e
        # print(temp)
        if(temp!=" " and temp!=""):
          finaltokens.append(temp)
      # tokens=finaltokens
    # print(finaltokens)
        

   
    # # a=[ch for ch in  if not ch in string.punctuation and ch not in ch.isdigit()] 
    # for each in tokens:
    #   temp=""
    #   # each =''.join([ch for ch in each if not ch in string.punctuation and int(ch) not  ch.isdigit()])
    #   for e in each:
    #     if(e not in string.punctuation) :
    #       temp+=e
    #   each=temp
      
    #   finaltokens.append(each)
    # tokens=finaltokens
    result=[]
    tokens=finaltokens
    wordnet = WordNetLemmatizer()
    for token,tag in pos_tag(tokens):
        pos=tag[0].lower()
        
        if pos not in ['a', 'r', 'n', 'v']:
            pos='n'
        

            
        result.append(wordnet.lemmatize(token,pos))
    
    tokens=result
    # stemming tokens
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(token) for token in tokens]


    # text=' '.join(text)
   
    # join tokens and form string
    preprocessed_text = " ".join(tokens)
    html_pattern = re.compile('<.*?>')
    preprocessed_text= html_pattern.sub(r'', preprocessed_text)
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    preprocessed_text=url_pattern.sub(r'', preprocessed_text)
    
    for word in preprocessed_text.split():
        if word not in finalwords:
            finalwords[word]=1
        else:
            finalwords[word]+=1
        


    return preprocessed_text

# sample text
# text = "Charles Babbage, who was born in 1791, is regarded as the father of computing because of his research into machines that could calculate."

# print("The preprocessed text of sample text is:", text_preprocessing(text), sep='\n')


# print("hello")
# for each in x[1]:
#     text_preprocessing(each)


def classify(text):
    if(text=="spam"):
        return 0
    
    else:
        return 1
    


x[1] = x[1].str.lower()
x[1]=x[1].apply(text_preprocessing )
x[0]=x[0].apply(classify)
x[[1]]



print(finalwords)
print(len(finalwords))




for k,v in finalwords.items():
    # print(k+str(v))
    finalwords[k]=np.log(5572 / (float(v) + 1))


print(finalwords)

list_of_dic=[]
dic={}
for ind in x.index:
    
    dic=dict.fromkeys(finalwords.keys(),0)
    # print(dic)
    count=0
    for word in x[1][ind].split():
        # print(word)
        dic[word]+=1
        count+=1
    # print(count)
    temp=[]
    for each in dic.keys():
      if(count!=0):
          dic[each]=((dic[each]/float(count))*finalwords[each])
          temp.append(dic[each])
    list_of_dic.append(np.array(temp))
    





# x[1].apply(starttf)


x[2]=list_of_dic
print("================================================")
# print(x)


# n=len(list_of_dic)

# print(finalwords)




x















print("hello")


print("==============================================")
print(type(x[0]))
print(x[0])
label_array = x[0].to_numpy()
print(label_array[0])
# tfidf_array = x[2].to_list() 
tfidf_array= np.array(x[2])
# print(tfidf_array[0])
# t_array = x[2].to_numpy()
# print(len(t_array))



print(type(label_array))


X_train, X_test, y_train, y_test = train_test_split(tfidf_array, label_array, test_size=0.25, random_state=0)

print(type(X_train))
print(len(X_train))
print(X_train[0])
print(len(y_train))
print(len(X_test))
print(len(y_test))
print(X_train[1000])
print(X_train[100])
if(len( X_train[1000])==0) :
  print("yes")
else:
  print("no")

print(x[0])
















def euclidean(b,a):
  # print(a)
  # a=np.array(a)
  # b=np.array(b)
  # print(type(a))
  return np.linalg.norm(a-b)


def manhattan(v1,v2):
    # print("hello")
    mandis=0
    # for ind in range(len(v1)):
    #     sq_dis=abs(v1[ind]-v2[ind])
    #     eucdis=eucdis+sq_dis
    mandis = (np.abs(v1 - v2)).sum()
    return mandis

def hamdis(v1,v2):
    return (manhattan(v1,v2)/len(finalwords))


def cosdis(v1,v2):
    # v1=np.array(v1)
    # v2=np.array(v2)
    result = 1-(dot(v1, v2)/(norm(v1)*norm(v2)))
    return result

# print("The Cosine Similarity between two vectors is: ",result)
    
    
    



#     lis=[]
#     tempeuc=[]
#         for k,v in train[3][trainind].items():
#             lis.append(v)
#     for trainind in train.index:
#         li=[]
#         for k,v in train[3][trainind].items():
#             li.append(v)
#         moretempeuc=[]
#         moretempeuc.append(euclidean(lis,li))
#         moretempeuc.append(train[0][trainind])
#         tempeuc.append(moretempeuc)
#     euclist.append(tempeuc)

        
                
    
        
    
        
        

    
    
    
    
# for testind in test.index:
#     lis=[]
#     tempman=[]
#         for k,v in train[3][trainind].items():
#             lis.append(v)
#     for trainind in train.index:
#         li=[]
#         for k,v in train[3][trainind].items():
#             li.append(v)
#         moretempman=[]
#         moretempman.append(euclidean(lis,li))
#         moretempman.append(train[0][trainind])
#         tempman.append(moretempman)
#     manlist.append(tempman)
    
    
    
# for testind in test.index:
#     lis=[]
#     tempham=[]
#         for k,v in train[3][trainind].items():
#             lis.append(v)
#     for trainind in train.index:
#         li=[]
#         for k,v in train[3][trainind].items():
#             li.append(v)
#         moretempham=[]
#         moretempham.append(euclidean(lis,li))
#         moretempham.append(train[0][trainind])
#         tempham.append(moretempham)
#     hamlist.append(tempham)
    
    

# for testind in test.index:
#     lis=[]
#     tempcos=[]
#         for k,v in train[3][trainind].items():
#             lis.append(v)
#     for trainind in train.index:
#         li=[]
#         for k,v in train[3][trainind].items():
#             li.append(v)
#         moretempcos=[]
#         moretempcos.append(euclidean(lis,li))
#         moretempcos.append(train[0][trainind])
#         tempcos.append(moretempcos)
#     coslist.append(tempcos)
    
        
        

        

        
        
        
        


manlist=[]
hamlist=[]
coslist=[]



f1value=[]
# for k in [1,3,5,7,11,17,23,28]:
# euclist=[]
tempeuclist=[]
count=0
# p=[]
# p=np.array(p)
# print(p.shape)
for c in range(len(X_test)):
  templis=[]
  
  for y in range(len(X_train)):
      moretemplis=[]
    
      if(len(X_train[y])!=0 and len(X_test[c])!=0):
        d=euclidean(X_train[y],X_test[c])
        moretemplis.append(d)
        moretemplis.append(y_train[y])
        templis.append(moretemplis)
  templis.sort()
  # print("Llen of templis is" + str(len(templis)))
  templis=templis[:30]
  tempeuclist.append(templis)
  # spam=0
  # ham=0
  # for each in templis:
  #   if( each[1]==0):

  #     spam+=1
  #   else:
  #     ham+=1
  # if(spam>ham):
  #   count+=1
  #   euclist.append(0)
  # else:
  #   count+=1
  #   euclist.append(1)
          
# print(euclist)
# print(len(euclist))

# # Accuracy
# from sklearn.metrics import accuracy_score
# print(accuracy_score(list(y_test), euclist))
# # Recall
# from sklearn.metrics import recall_score
# print(recall_score(list(y_test), euclist, average=None))
# # Precision


    # print(count)

    # print(kvalue)
    # print(f1value)




# for x in range(len(X_test)):
#   # count+=1
#   # if(count<10):
#   #   print("the vector is:" )
#   #   y=np.array(x)
#   #   print(type(y))
#   #   print(y.shape)

#   # else:
#   #   break
#     templis=[]
# # print(type(test[3][testind]))
#     # for k,v in test[3][testind].items():
#     #     lis.append(v)
#     for y in range(len(X_train)):
#         # li=[]
       
#         # print(count)
        
#         #   # print("the vector is:" )
#         #   # y=np.array
#         #   # print(type(y))
        
#         # for k,v in train[3][trainind].items():
#         #     li.append(v)
#         moretemplis=[]
#         # if(count==1001):
#         #   print(X_train[y])
#         #   print(y)
#         #   print(X_test[x])
#         if(len(X_train[y])!=0 and len(X_test[x])!=0):
#           d=cosdis(X_train[y],X_test[x])
          
#           moretemplis.append(d)
#           moretemplis.append(y_train[y])
#           templis.append(moretemplis)
#           # print(templis)

#     templis.sort()
#     templis=templis[:9]
#     # count+=1
#     #       if(count<3):
#     #         print()
#     print(templis)
#     spam=0;
#     ham=0
#     for each in templis:
#         if( each[1]==0):
#             spam+=1
#         else:
#             ham+=1
#     # print(spam)
#     # print(ham)
#     if(spam>ham):
#         coslist.append(0)
#     else:
#         coslist.append(1)
            
# print(coslist)



# print(manlist)

# for x in range(len(X_test)):
#   # count+=1
#   # if(count<10):
#   #   print("the vector is:" )
#   #   y=np.array(x)
#   #   print(type(y))
#   #   print(y.shape)

#   # else:
#   #   break
#     templis=[]
# # print(type(test[3][testind]))
#     # for k,v in test[3][testind].items():
#     #     lis.append(v)
#     for y in range(len(X_train)):
#         # li=[]
#         # count+=1
#         # print(count)
#         # if(count<3):
#         #   print(euclidean(X_train[y],X_test[x]))
#         #   # print("the vector is:" )
#         #   # y=np.array
#         #   # print(type(y))
        
#         # for k,v in train[3][trainind].items():
#         #     li.append(v)
#         moretemplis=[]
#         # if(count==1001):
#         #   print(X_train[y])
#         #   print(y)
#         #   print(X_test[x])
#         if(len(X_train[y])!=0 and len(X_test[x])!=0):
#           d=manhattan(X_train[y],X_test[x])
#           moretemplis.append(d)
#           moretemplis.append(y_train[y])
#           templis.append(moretemplis)
#     templis.sort()
#     templis=templis[:5]
#     spam=0;
#     ham=0
#     for each in templis:
#         if( each[1]==0):
#             spam+=1
#         else:
#             ham+=1
#     if(spam>ham):
#         manlist.append(0)
#     else:
#         manlist.append(1)
            
# print(manlist)



# for x in range(len(X_test)):
#   # count+=1
#   # if(count<10):
#   #   print("the vector is:" )
#   #   y=np.array(x)
#   #   print(type(y))
#   #   print(y.shape)

#   # else:
#   #   break
#     templis=[]
# # print(type(test[3][testind]))
#     # for k,v in test[3][testind].items():
#     #     lis.append(v)
#     for y in range(len(X_train)):
#         # li=[]
#         # count+=1
#         # print(count)
#         # if(count<3):
#         #   print(euclidean(X_train[y],X_test[x]))
#         #   # print("the vector is:" )
#         #   # y=np.array
#         #   # print(type(y))
        
#         # for k,v in train[3][trainind].items():
#         #     li.append(v)
#         moretemplis=[]
#         # if(count==1001):
#         #   print(X_train[y])
#         #   print(y)
#         #   print(X_test[x])
#         if(len(X_train[y])!=0 and len(X_test[x])!=0):
#           d=hamdis(X_train[y],X_test[x])
#           moretemplis.append(d)
#           moretemplis.append(y_train[y])
#           templis.append(moretemplis)
#     templis.sort()
#     templis=templis[:5]
#     spam=0;
#     ham=0
#     for each in templis:
#         if( each[1]==0):
#             spam+=1
#         else:
#             ham+=1
#     if(spam>ham):
#         hamlist.append(0)
#     else:
#         hamlist.append(1)
            
# print(hamlist)


manlist=[]
tempmanlist=[]
count=0
# p=[]
# p=np.array(p)
# print(p.shape)
for c in range(len(X_test)):
  templis=[]
  
  for y in range(len(X_train)):
      moretemplis=[]
    
      if(len(X_train[y])!=0 and len(X_test[c])!=0):
        d=manhattan(X_train[y],X_test[c])
        moretemplis.append(d)
        moretemplis.append(y_train[y])
        templis.append(moretemplis)
  templis.sort()
  # print("Llen of templis is" + str(len(templis)))
  templis=templis[:30]
  tempmanlist.append(templis)







# # coslist=[]
tempcoslist=[]
count=0

for c in range(len(X_test)):
  templis=[]
  
  for y in range(len(X_train)):
      moretemplis=[]
    
      if(len(X_train[y])!=0 and len(X_test[c])!=0):
        d=cosdis(X_train[y],X_test[c])
        moretemplis.append(d)
        moretemplis.append(y_train[y])
        templis.append(moretemplis)
  templis.sort()
  # print("Llen of templis is" + str(len(templis)))
  templis=templis[:30]
  tempcoslist.append(templis)



# hamlist=[]
temphamlist=[]
count=0

for c in range(len(X_test)):
  templis=[]
  
  for y in range(len(X_train)):
      moretemplis=[]
    
      if(len(X_train[y])!=0 and len(X_test[c])!=0):
        d=hamdis(X_train[y],X_test[c])
        moretemplis.append(d)
        moretemplis.append(y_train[y])
        templis.append(moretemplis)
  templis.sort()
  # print("Llen of templis is" + str(len(templis)))
  templis=templis[:30]
  temphamlist.append(templis)




f1value=[]
for k in [1,3,5,7,11,17,23,28]:
  # kvalue.append(k)
  
  euclist=[]
  for templis in tempeuclist:
    spam=0
    ham=0
    templis=templis[:k]
    for each in templis:
      if( each[1]==0):

        spam+=1
      else:
        ham+=1
    if(spam>ham):
      count+=1
      euclist.append(0)
    else:
      count+=1
      euclist.append(1)
  fig, ax = plt.subplots(figsize=(5, 5))
  f1value.append(f1_score(list(y_test), euclist, average="macro"))
  print("the accuracy score for euclidean is " +str(accuracy_score(list(y_test), euclist)))
  confmat=confusion_matrix(list(y_test), euclist)
  ax.matshow(confmat, cmap=plt.cm.Oranges, alpha=0.3)
  for i in range(confmat.shape[0]):
      for j in range(confmat.shape[1]):
          ax.text(x=j, y=i,s=confmat[i, j], va='center', ha='center', size='xx-large')

  plt.xlabel('Predictions', fontsize=18)
  plt.ylabel('Actuals', fontsize=18)
  plt.title('Confusion Matrix  Euclidean for k = ' + str(k), fontsize=18)
  plt.show()

  print(classification_report(y_test, euclist))


print(f1value)
plt.plot([1,3,5,7,11,17,23,28], f1value)
 
# naming the x axis
plt.xlabel('k-value')
# naming the y axis
plt.ylabel('f1 - score')
 
# giving a title to my graph
plt.title(' graph plotting f1sore value vs k!')
 
# function to show the plot
plt.show()

print(euclist)
print(len(euclist))








f1value=[]
for k in [1,3,5,7,11,17,23,28]:
  # kvalue.append(k)
  
  manlist=[]
  for templis in tempmanlist:
    spam=0
    ham=0
    templis=templis[:k]
    for each in templis:
      if( each[1]==0):

        spam+=1
      else:
        ham+=1
    if(spam>ham):
      count+=1
      manlist.append(0)
    else:
      count+=1
      manlist.append(1)
  fig, ax = plt.subplots(figsize=(5, 5))
  f1value.append(f1_score(list(y_test), manlist, average="macro"))
  print("the accuracy score for manhattan is " +str(accuracy_score(list(y_test), manlist)))
  confmat=confusion_matrix(list(y_test), manlist)
  ax.matshow(confmat, cmap=plt.cm.Oranges, alpha=0.3)
  for i in range(confmat.shape[0]):
      for j in range(confmat.shape[1]):
          ax.text(x=j, y=i,s=confmat[i, j], va='center', ha='center', size='xx-large')

  plt.xlabel('Predictions', fontsize=18)
  plt.ylabel('Actuals', fontsize=18)
  plt.title('Confusion Matrix  for manhattan k = ' + str(k), fontsize=18)
  plt.show()

  print(classification_report(y_test, manlist))


print(f1value)
plt.plot([1,3,5,7,11,17,23,28], f1value)
 
# naming the x axis
plt.xlabel('k-value')
# naming the y axis
plt.ylabel('f1 - score')
 
# giving a title to my graph
plt.title(' graph plotting f1sore value vs k!')
 
# function to show the plot
plt.show()







f1value=[]
for k in [1,3,5,7,11,17,23,28]:
  # kvalue.append(k)
  
  coslist=[]
  for templis in tempcoslist:
    spam=0
    ham=0
    templis=templis[:k]
    for each in templis:
      if( each[1]==0):

        spam+=1
      else:
        ham+=1
    if(spam>ham):
      count+=1
      coslist.append(0)
    else:
      count+=1
      coslist.append(1)
  fig, ax = plt.subplots(figsize=(5, 5))
  f1value.append(f1_score(list(y_test), coslist, average="macro"))
  print("the accuracy score for cosine is " +str(accuracy_score(list(y_test), coslist)))
  confmat=confusion_matrix(list(y_test), coslist)
  ax.matshow(confmat, cmap=plt.cm.Oranges, alpha=0.3)
  for i in range(confmat.shape[0]):
      for j in range(confmat.shape[1]):
          ax.text(x=j, y=i,s=confmat[i, j], va='center', ha='center', size='xx-large')

  plt.xlabel('Predictions', fontsize=18)
  plt.ylabel('Actuals', fontsize=18)
  plt.title('Confusion Matrix  for  cosine k = ' + str(k), fontsize=18)
  plt.show()

  print(classification_report(y_test, coslist))


print(f1value)
plt.plot([1,3,5,7,11,17,23,28], f1value)
 
# naming the x axis
plt.xlabel('k-value')
# naming the y axis
plt.ylabel('f1 - score')
 
# giving a title to my graph
plt.title(' graph plotting f1sore value vs k!')
 
# function to show the plot
plt.show()







f1value=[]
for k in [1,3,5,7,11,17,23,28]:
  # kvalue.append(k)
  
  hamlist=[]
  for templis in temphamlist:
    spam=0
    ham=0
    templis=templis[:k]
    for each in templis:
      if( each[1]==0):

        spam+=1
      else:
        ham+=1
    if(spam>ham):
      count+=1
      hamlist.append(0)
    else:
      count+=1
      hamlist.append(1)
  fig, ax = plt.subplots(figsize=(5, 5))
  f1value.append(f1_score(list(y_test), hamlist, average="macro"))
  print("the accuracy score for hamming  is " +str(accuracy_score(list(y_test), hamlist)))
  confmat=confusion_matrix(list(y_test), hamlist)
  ax.matshow(confmat, cmap=plt.cm.Oranges, alpha=0.3)
  for i in range(confmat.shape[0]):
      for j in range(confmat.shape[1]):
          ax.text(x=j, y=i,s=confmat[i, j], va='center', ha='center', size='xx-large')

  plt.xlabel('Predictions', fontsize=18)
  plt.ylabel('Actuals', fontsize=18)
  plt.title('Confusion Matrix  for hamming  k = ' + str(k), fontsize=18)
  plt.show()

  print(classification_report(y_test, hamlist))


print(f1value)
plt.plot([1,3,5,7,11,17,23,28], f1value)
 
# naming the x axis
plt.xlabel('k-value')
# naming the y axis
plt.ylabel('f1 - score')
 
# giving a title to my graph
plt.title(' graph plotting f1sore value vs k!')
 
# function to show the plot
plt.show()





#creating the tfidf vectors
# x
# from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(x[1])
print(vectors)
features = vectors
X_train, X_test, y_train, y_test = train_test_split(features, x[0], test_size=0.15, random_state=111)
knn = KNeighborsClassifier(n_neighbors=7,metric='cosine',
                         algorithm='brute',
                         n_jobs=-1)
# model = NearestNeighbors(n_neighbors=n_neighbor,
#                          metric='cosine',
#                          algorithm='brute',
#                          n_jobs=-1)
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


















