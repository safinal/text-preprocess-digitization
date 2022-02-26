from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def main(text_file_path: str) -> None:

    # Open and read the text file and store it in a constant variable 
    with open(text_file_path, 'r') as f:
        SAMPLE_TEXT = f.read()

    ALLOWED_CHARACTERS = "abcdefghijklmnopqrstuvwxyz -"

    cleaned_text = text_to_single_paragraph(text=SAMPLE_TEXT)

    # Convert the whole text characters to lower-case form
    cleaned_text = cleaned_text.lower()

    cleaned_text = remove_unallowed_characters(text=cleaned_text, alphabet=ALLOWED_CHARACTERS)

    word_tokens = tokenize_text(text=cleaned_text)
    print(f'\nNumber of words BEFORE removing stop words and stemming: {len(word_tokens)}')

    filtered_word_tokens = remove_stop_words(word_tokens=word_tokens)

    stemmed_word_tokens = stem_word_tokens(word_tokens=filtered_word_tokens)
    print(f'Number of words AFTER removing stop words and stemming: {len(stemmed_word_tokens)}\n')

    for token in stemmed_word_tokens:
        vectors = onehot_w2v(word=token, alphabet=ALLOWED_CHARACTERS)
        pretty_print(word=token, vectors=vectors)


def text_to_single_paragraph(text: str) -> str:
    # Convert the whole text to one paragraph
    return text.replace('\n', ' ')


def remove_unallowed_characters(text: str, alphabet: str) -> str:
    temp = text
    for char in text:
        if char not in alphabet:
            temp = temp.replace(char, '')
    return temp


def tokenize_text(text: str) -> set:
    """
    Split the text into a set of words (Tokenization)
    """
    return set(word_tokenize(text))


def remove_stop_words(word_tokens: set) -> list:
    return [token for token in word_tokens if token not in stopwords.words('english') and len(token) > 1]


def stem_word_tokens(word_tokens: list) -> list:
    """
    Using Porter stemmer to stem all words
    """
    porter = PorterStemmer()
    tokens = set()
    for token in word_tokens:
        tokens.add(porter.stem(token))
    return list(tokens)


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
