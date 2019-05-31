from bs4 import BeautifulSoup as BS
from urllib.request import urlopen
import json

html = urlopen("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=20190531")
soup = BS(html)
rows = soup.findAll("tr", class_="text-right")
print(rows[0].findAll("td"))

separated_rows = []
for row in rows:
    separated_rows.append(row.findAll("td"))

cleaned_data = []
for row in separated_rows:
    clRow = {}
    clRow["date"] = row[0].contents
    clRow["open"] = row[1].contents
    clRow["high"] = row[2].contents
    clRow["low"] = row[3].contents
    clRow["close"] = row[4].contents
    clRow["volume"] = row[5].contents
    clRow["mark_cap"] = row[6].contents
    cleaned_data.append(clRow)

print(cleaned_data)

with open('historical_data.json', 'w') as outfile:
    outfile.write(json.dumps(cleaned_data))


