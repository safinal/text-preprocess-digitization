from nltk.stem import PorterStemmer


def main(text_file_path: str) -> None:

    # Open and read the text file and store it in a constant variable 
    with open(text_file_path, 'r') as f:
        SAMPLE_TEXT = f.read()

    ALLOWED_CHARACTERS = 'abcdefghijklmnopqrstuvwxyz '

    cleaned_text = clean_text(text=SAMPLE_TEXT, alphabet=ALLOWED_CHARACTERS)

    words = tokenize_text(text=cleaned_text)

    for word in words:
        vectors = onehot_w2v(word=word, alphabet=ALLOWED_CHARACTERS)
        pretty_print(word=word, vectors=vectors)


def clean_text(text: str, alphabet: str) -> str:

    # Convert the whole text to one paragraph
    text = text.replace('\n', ' ')

    # Convert the whole text characters to lower-case form
    text = text.lower()

    # Remove unallowed characters from the text
    temp = text
    for char in text:
        if char not in alphabet:
            temp = temp.replace(char, '')
    return temp


def tokenize_text(text: str) -> set:
    """
    Split the text into a set of words (Tokenization) 
    and using Porter stemmer to stem all words
    """
    porter = PorterStemmer()
    words = set()
    for word in [word for word in text.split() if len(word) > 1]:
        words.add(porter.stem(word))
    return words


def onehot_w2v(word: str, alphabet: str) -> list:
    """
    Converts a given word of type string to the one-hot 
    representation of the same word (list of vectors)
    """
    temp_list = []
    for letter in word:
        temp_list.append(alphabet.find(letter))
    vectors_list = [[0]*len(alphabet) for _ in range(len(word))]

    for i in range(len(word)):
        vectors_list[i][temp_list[i]] = 1

    return vectors_list


def pretty_print(word: str, vectors: list) -> None:
    print(f'{word}:\n[')
    for vector in vectors:
        print(f'    {vector},')
    print(']\n')


main(text_file_path=r'F:\uni\7_4002\Foundations of Information Retrieval and Web Search\Homeworks\sample_text.txt')
