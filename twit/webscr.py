import dryscrape
from bs4 import BeautifulSoup
from bs4 import Comment
session = dryscrape.Session()
session.visit("https://www.nur.kz/1749446-glavu-otdela-kultury-nakazali-za-grubost-v-karagandinskoj-oblasti.html")
response = session.body()
#print (response)
soup=BeautifulSoup(response,'html.parser')
for comments in soup.find_all(string=lambda html_comment:isinstance(html_comment, Comment)):
    print(comments.extract())
