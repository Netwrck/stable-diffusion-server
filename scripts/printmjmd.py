# open converted.csv
import csv
from urllib.parse import quote

md_links = []
with open('converted.csv', 'r') as f:
    reader = csv.reader(f)
    converted = list(reader)
    for row in converted:
        # convert to a link
        prompt = row[0]
        link = prompt#.replace(" ", "%20")
        urlpart_link = quote(link)

        md_link = f"[{prompt}](https://www.ebank.nz/aiartgenerator?category={urlpart_link})"
        #print(md_link)
        md_links.append(md_link)


# write each 100 to a different md file called ai-art-generatorN.md

# write a table of contents

for i in range(0, len(md_links), 100):
    for j in range(i, i + 100):
        with open(f"ai-art-generator{i}.md", 'a') as f:
            f.write(md_links[j])
