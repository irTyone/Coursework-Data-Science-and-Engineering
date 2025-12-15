import jieba
with open('stopwords.txt','r',encoding='utf-8') as f:
     text = f.read()
print(type(text))
# words=jieba.lcut(text)
# print(type(words))
temp=''
words=[]
for token in text:
    if token != '\n':
        temp+=token
    else :
        words.append(temp)
        temp=''

for i ,word in enumerate(words):
     print(f"{i}:{word}")