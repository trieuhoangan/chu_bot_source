link = "QA.txt"
Q = []
A = []
with open(link,'r',encoding='utf-8') as fin:
    lines = fin.readlines()
    for line in lines:
        QnA  = line.split('@@@@@')
        Q.append(QnA[0])
        if len(QnA) >1:
            A.append(QnA[1])

with open('Q.txt','w',encoding='utf-8') as foutQ:
    for single_Q in Q:
        foutQ.write(single_Q+'\n')
with open('A.txt','w',encoding='utf-8') as foutA:
    for single_A in A:
        foutA.write(single_A)