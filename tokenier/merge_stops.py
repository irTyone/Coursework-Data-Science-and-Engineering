import os
import json
from socket import if_nameindex
TXT_DIR="/home/liuyuan/Class_data/vocab"
SAVE_PATH='/home/liuyuan/Class_data/vocab/stop_merge.json'

def merges_stop_words()->set:
    stop_set=set()
    for filename in os.listdir(TXT_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(TXT_DIR,filename),'r',encoding='utf-8')as f:
                file=f.read()
                stops_part=get_list_words(file)
                stop_set.update(stops_part)     
        else:
            continue
    return stop_set

def get_list_words(file:str):
    stop_list=[]
    temp=''
    for letter in file:
        if letter!='\n':
            temp+=letter
        else:
            stop_list.append(temp)
            temp=''             
    return set(stop_list)



def json_write(stops :set):
    dic={}
    dic['stops']=[]
    for _ in stops:
        dic.get("stops").append(_)
    with open('/home/liuyuan/Class_data/vocab/stop_merge.json','w',encoding='utf-8') as f:
        json.dump(dic,f,ensure_ascii=False,indent=2)
        
        
def main():
    stops=merges_stop_words()
    json_write(stops)



if __name__ == "__main__":
    main()
