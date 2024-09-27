import unicodedata
import torch



class GenericTokenizer:
    def __init__(self, **kwargs):
        [setattr(self, key, value) for key, value in kwargs.items()] # Arbitrarily accept all keywords passed
 
    def train(self, **kwargs):
        raise NotImplementedError

    def encode(self, input: str) -> list:
        """
        Inputs a string, returns a list of integers, i.e. the token encodings
        """
        raise NotImplementedError

    def decode(self, input: list) -> str:
        """
        Inputs a list of integers, returns a string
        List of integers is the generated token list
        """
        raise NotImplementedError  

    def save(self, file_name: str) -> None:
        """
        Saves two files: file_name.model and file_name.vocab
        These can be used for loading and visualizing the model (e.g. the vocab + TODO) respectively
        .model can be used for the load() function
        """
        raise NotImplementedError
    
    def load(self, file_name: str):
        """
        Loads file_name.model, file_name has to be a direct reference to the correct .model file
        May work with .model files not saved by my code? TODO TEST
        """
        raise NotImplementedError

    def forward_padding_mask(self, generic_tensor: torch.Tensor, padding_mask: torch.Tensor,) -> torch.Tensor:
        """
        Padding masking is maskeren (dus het niet meegeven aan het model) van de extra lege ruimte die je toevoegt om iets de grootte van je batch size te maken
        e.g.
        Tensor([[1, 22, 3, 0, 0, 0],
        [2, 132, 55, 14, 21, 6],
        2, 1, 8, 44, 5, 0]])
        Padding masking is dus het 'verdwijnen' van de nul-paddings.
        generic_tensor is een tensor van een var lengte met mogelijk padding en mogelijke daadwerlijke informatie
        padding_mask is de 'opslag' van welke stukjes data van de generic_tensor tensor echt zijn (1) en welke padding (0)
        Let dus dat deze functie data agnostisch is, zolang gen_tensor en padding_mask dezelfde dimensionaliteit hebben, komt het altijd goed
        """
        extra = padding_mask.size(1) % generic_tensor.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), generic_tensor.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask


# -----------------------------
# Generic helper functions TODO Figure out if I can just move these to inside the class -> why?:

def count_pairs(intList: list, counts=None) -> dict:
    """
    Given a list of itegers, return a dict of the counts of consecutive pairs of integers
    e.g. [1,2,3,1,2] ->[(1,2): 2, (2,3): 1, (3,1): 1]
    Can use an existing counts dictionary, never used that before though
    """
    counts = {} if counts is None else counts
    for pair in zip(intList, intList[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(intList: list, pair, replace: int):
    """
    In the list of integers, replace all consecutive occurences
    of pair with the new integer token (i.e. the replace value)
    e.g. ids=[1,2,3,1,2]. pair = (1,2), idx = 4 -returns> [4,3,4]
    """

    """
    Dit werkt door langs waarde I van intList te loopen en te checken of 
    waarde i en waarde i + 1 (de volgende waarde dus) samen gelijk zijn aan 'pair'.
    Is dit het geval, voeg dan replace toe aan newIntList en spring door naar i + 2
    (i.e. het getal net na het paar wat je net teste) Dit betekent dat i en i+1 niet aan 
    newIntList worden toegevoegd, maar replace wel
    Matchen ze niet, plak dan gwn i aan de nieuwe lijst en ga naar het volgende getal. 
    """
    newIntList = []
    i = 0
    while i < len(intList):
        # if the pair matches
        if intList[i] == pair[0] and intList[i+1] == pair[1]:
            # The if statement can occasionally fail if pair = (x, None) and intList ends with x, thus:
            if i < len(intList) - 1:
                newIntList.append(replace)
                i += 2
        else:   
            newIntList.append(intList[i])
            i += 1

def replace_control_chars(s: str) -> str:
    """
    Replaces string specific chars (\n)
    """
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
            # het karakter wat NIET in de C category van unicode valt
            # is een veilig karakter, daar gaan we mee doer
        else:
            chars.append(f"\\u{ord(ch):04x}")
            # een escape karakter, e.g. \ in regex
        return "".join(chars)

def render_token(t: bytes) -> str:
    """
    Returns a renderable string of the given bytes object, with control chars escaped
    """
    s = t.decode("utf-8", errors="replace")
    s = replace_control_chars(s)
    return s

















text = "AAAABBBCCC"

chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create mapping from the characters=>integers
stringtoint = { ch:i for i,ch in enumerate(chars) }
inttostring= { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stringtoint[c] for c in s]
decode = lambda l: ''.join([inttostring[i] for i in l])
