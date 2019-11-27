def lowerall(link_file):
    
    with open(link_file,'r',encoding='utf-8') as fin:
        rows=fin.readlines()
    fout = open('a.txt','w',encoding='utf-8')
    for row in rows:
        fout.write(row.lower())
    fout.close()
if __name__=="__main__":
    link_file="a.txt"
    lowerall(link_file)