

import os
import json

with open("Code/datasets/wouter.json", 'r') as f:
    text = json.load(f)
    f.close()

new = ""
for i in text:
    for key, value in i.items():
        new += key + "\n" + value

print(new)

with open("Code/datasets/wouter.txt", 'w') as d:
    d.write(new)