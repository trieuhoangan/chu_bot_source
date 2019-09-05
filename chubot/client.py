import requests


ip = "192.168.43.56"
mess = "xin chao"
r = requests.get("http://"+ip+"/?mess="+mess)

print(r.json()["response"][0])
