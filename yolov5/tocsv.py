from numpy import append, double
import pandas as pd
import os

root='F:\\test_mission\\src\\yolov5\\runs\\detect\\exp11\\labels'#记得下一次改这里的路径
lst1=[]
lst2=[]
for i in range(40000):
    s=os.path.join(root,"%06d"%i)
    lst1.append("%06d"%i+'.png')
    if os.path.exists(s+'.txt'):
        lin=""
        strlst=[]
        flst=[]
        with open(s+'.txt', 'r') as file1:
            for line in file1.readlines():
                id,x,*parm=line.strip().split(' ')
                val=float(x)
                flag=0
                for i in range(len(flst)):
                    if val<flst[i]:
                        flag=1
                        strlst.insert(i,id)
                        break
                flst.append(val)
                flst.sort()
                if flag==0:
                    strlst.append(id)
        for u in strlst:
            lin=lin+u
        lst2.append(lin)
    else:
        lst2.append('1')#1最容易被识别成背景
#print(lst2)
csvroot='./submission-yolov5x.csv'
df= pd.DataFrame({'file_name':lst1,'file_code':lst2})
df.to_csv(csvroot,index=False)
