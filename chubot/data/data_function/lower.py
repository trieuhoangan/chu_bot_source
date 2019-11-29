from changeUnicode import ChangeUniCode
def lowerall(link_file):
    changer = ChangeUniCode()
    with open(link_file,'r',encoding='utf-8') as fin:
        rows=fin.readlines()
    fout = open('a.txt','w',encoding='utf-8')
    for row in rows:
        row = changer.compound_unicode(row)
        fout.write(row.lower())
    fout.close()
def change_all_Unicode(linkfile):
    with open(link_file,'r',encoding='utf-8') as fin:
        rows=fin.readlines()
    fout = open('a.txt','w',encoding='utf-8')
    for row in rows:
        fout.write(row.lower())
    fout.close()
if __name__=="__main__":
    link_file="a.txt"
    lowerall(link_file)