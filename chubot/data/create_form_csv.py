
Q_link = 'Q.txt'
A_link = 'A.txt'

with open(Q_link,'r',encoding='utf-8') as Qin:
    rows = Qin.readlines()

for row in rows:
    row = row +',,'