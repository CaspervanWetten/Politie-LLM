# File to keep the GPT file clean, customs from the video to suit my needs



from os import getcwd, path
r"""gets input.txt if not from this datasets folder, from the datasets folder in ../Code
"""
def get_input(inp = "shakespeare"):
    if inp == "shakespeare":
        try:
            with open('datasets/input.txt', 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            with open('Code/datasets/input.txt', 'r', encoding='utf-8') as f:
                text = f.read()
        return text

    if inp == "wouter":
        try:
            with open('datasets/wouter.txt', 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            with open('Code/datasets/wouter.txt', 'r', encoding='utf-8') as f:
                text = f.read()
        return text
    
    raise FileNotFoundError("Bestand niet gevonden!")