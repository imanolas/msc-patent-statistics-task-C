import nltk
import unidecode
from bs4 import BeautifulSoup as bs

nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer

# the stemmer requires a language parameter
snow_stemmer_en = SnowballStemmer(language='english')
snow_stemmer_fr = SnowballStemmer(language='french')
snow_stemmer_de = SnowballStemmer(language='german')

stop_words_en = set(nltk.corpus.stopwords.words('english'))
stop_words_fr = set(nltk.corpus.stopwords.words('french'))
stop_words_de = set(nltk.corpus.stopwords.words('german'))

tokenizer = nltk.RegexpTokenizer(r'\w+')

lemmatizer = nltk.stem.WordNetLemmatizer()

german_chars = {b'\xc3\x80': 'À', b'\xc3\x81': 'Á', b'\xc3\x82': 'Â', b'\xc3\x83': 'Ã', b'\xc3\x84': 'Ä',
                b'\xc3\x85': 'Å',
                b'\xc3\x86': 'Æ', b'\xc3\x87': 'Ç', b'\xc3\x88': 'È', b'\xc3\x89': 'É', b'\xc3\x8a': 'Ê',
                b'\xc3\x8b': 'Ë',
                b'\xc3\x8c': 'Ì', b'\xc3\x8d': 'Í', b'\xc3\x8e': 'Î', b'\xc3\x8f': 'Ï', b'\xc3\x90': 'Ð',
                b'\xc3\x91': 'Ñ',
                b'\xc3\x92': 'Ò', b'\xc3\x93': 'Ó', b'\xc3\x94': 'Ô', b'\xc3\x95': 'Õ', b'\xc3\x96': 'Ö',
                b'\xc3\x97': '×',
                b'\xc3\x98': 'Ø', b'\xc3\x99': 'Ù', b'\xc3\x9a': 'Ú', b'\xc3\x9b': 'Û', b'\xc3\x9c': 'Ü',
                b'\xc3\x9d': 'Ý',
                b'\xc3\x9e': 'Þ', b'\xc3\x9f': 'ß', b'\xc3\xa0': 'à', b'\xc3\xa1': 'á', b'\xc3\xa2': 'â',
                b'\xc3\xa3': 'ã',
                b'\xc3\xa4': 'ä', b'\xc3\xa5': 'å', b'\xc3\xa6': 'æ', b'\xc3\xa7': 'ç', b'\xc3\xa8': 'è',
                b'\xc3\xa9': 'é',
                b'\xc3\xaa': 'ê', b'\xc3\xab': 'ë', b'\xc3\xac': 'ì', b'\xc3\xad': 'í', b'\xc3\xae': 'î',
                b'\xc3\xaf': 'ï',
                b'\xc3\xb0': 'ð', b'\xc3\xb1': 'ñ', b'\xc3\xb2': 'ò', b'\xc3\xb3': 'ó', b'\xc3\xb4': 'ô',
                b'\xc3\xb5': 'õ',
                b'\xc3\xb6': 'ö', b'\xc3\xb7': '÷', b'\xc3\xb8': 'ø', b'\xc3\xb9': 'ù', b'\xc3\xba': 'ú',
                b'\xc3\xbb': 'û',
                b'\xc3\xbc': 'ü', b'\xc3\xbd': 'ý', b'\xc3\xbe': 'þ', b'\xc3\xbf': 'ÿ'}

german_chars2 = {b'\\xc3\\x80': 'À', b'\\xc3\\x81': 'Á', b'\\xc3\\x82': 'Â', b'\\xc3\\x83': 'Ã', b'\\xc3\\x84': 'Ä',
                 b'\\xc3\\x85': 'Å',
                 b'\\xc3\\x86': 'Æ', b'\\xc3\\x87': 'Ç', b'\\xc3\\x88': 'È', b'\\xc3\\x89': 'É', b'\\xc3\\x8a': 'Ê',
                 b'\\xc3\\x8b': 'Ë',
                 b'\\xc3\\x8c': 'Ì', b'\\xc3\\x8d': 'Í', b'\\xc3\\x8e': 'Î', b'\\xc3\\x8f': 'Ï', b'\\xc3\\x90': 'Ð',
                 b'\\xc3\\x91': 'Ñ',
                 b'\\xc3\\x92': 'Ò', b'\\xc3\\x93': 'Ó', b'\\xc3\\x94': 'Ô', b'\\xc3\\x95': 'Õ', b'\\xc3\\x96': 'Ö',
                 b'\\xc3\\x97': '×',
                 b'\\xc3\\x98': 'Ø', b'\\xc3\\x99': 'Ù', b'\\xc3\\x9a': 'Ú', b'\\xc3\\x9b': 'Û', b'\\xc3\\x9c': 'Ü',
                 b'\\xc3\\x9d': 'Ý',
                 b'\\xc3\\x9e': 'Þ', b'\\xc3\\x9f': 'ß', b'\\xc3\\xa0': 'à', b'\\xc3\\xa1': 'á', b'\\xc3\\xa2': 'â',
                 b'\\xc3\\xa3': 'ã',
                 b'\\xc3\\xa4': 'ä', b'\\xc3\\xa5': 'å', b'\\xc3\\xa6': 'æ', b'\\xc3\\xa7': 'ç', b'\\xc3\\xa8': 'è',
                 b'\\xc3\\xa9': 'é',
                 b'\\xc3\\xaa': 'ê', b'\\xc3\\xab': 'ë', b'\\xc3\\xac': 'ì', b'\\xc3\\xad': 'í', b'\\xc3\\xae': 'î',
                 b'\\xc3\\xaf': 'ï',
                 b'\\xc3\\xb0': 'ð', b'\\xc3\\xb1': 'ñ', b'\\xc3\\xb2': 'ò', b'\\xc3\\xb3': 'ó', b'\\xc3\\xb4': 'ô',
                 b'\\xc3\\xb5': 'õ',
                 b'\\xc3\\xb6': 'ö', b'\\xc3\\xb7': '÷', b'\\xc3\\xb8': 'ø', b'\\xc3\\xb9': 'ù', b'\\xc3\\xba': 'ú',
                 b'\\xc3\\xbb': 'û',
                 b'\\xc3\\xbc': 'ü', b'\\xc3\\xbd': 'ý', b'\\xc3\\xbe': 'þ', b'\\xc3\\xbf': 'ÿ', b'\\xc2\\x80': '',
                 b'\\xc2\\x81': '',
                 b'\\xc2\\x82': '', b'\\xc2\\x83': '', b'\\xc2\\x84': '', b'\\xc2\\x85': '', b'\\xc2\\x86': '',
                 b'\\xc2\\x87': '',
                 b'\\xc2\\x88': '', b'\\xc2\\x89': '', b'\\xc2\\x8a': '', b'\\xc2\\x8b': '', b'\\xc2\\x8c': '',
                 b'\\xc2\\x8d': '',
                 b'\\xc2\\x8e': '', b'\\xc2\\x8f': '', b'\\xc2\\x90': '', b'\\xc2\\x91': '', b'\\xc2\\x92': '',
                 b'\\xc2\\x93': '',
                 b'\\xc2\\x94': '', b'\\xc2\\x95': '', b'\\xc2\\x96': '', b'\\xc2\\x97': '', b'\\xc2\\x98': '',
                 b'\\xc2\\x99': '',
                 b'\\xc2\\x9a': '', b'\\xc2\\x9b': '', b'\\xc2\\x9c': '', b'\\xc2\\x9d': '', b'\\xc2\\x9e': '',
                 b'\\xc2\\x9f': '',
                 b'\\xc2\\xa0': '', b'\\xc2\\xa1': '', b'\\xc2\\xa2': '', b'\\xc2\\xa3': '', b'\\xc2\\xa4': '',
                 b'\\xc2\\xa5': '',
                 b'\\xc2\\xa6': '', b'\\xc2\\xa7': '', b'\\xc2\\xa8': '', b'\\xc2\\xa9': '', b'\\xc2\\xaa': '',
                 b'\\xc2\\xab': '',
                 b'\\xc2\\xac': '', b'\\xc2\\xad': '', b'\\xc2\\xae': '', b'\\xc2\\xaf': '', b'\\xc2\\xb0': '',
                 b'\\xc2\\xb1': '',
                 b'\\xc2\\xb2': '', b'\\xc2\\xb3': '', b'\\xc2\\xb4': '', b'\\xc2\\xb5': '', b'\\xc2\\xb6': '',
                 b'\\xc2\\xb7': '',
                 b'\\xc2\\xb8': '', b'\\xc2\\xb9': '', b'\\xc2\\xba': '', b'\\xc2\\xbb': '', b'\\xc2\\xbc': '',
                 b'\\xc2\\xbd': '',
                 b'\\xc2\\xbe': '', b'\\xc2\\xbf': ''}


def myreplace(match):
    try:
        if match:
            textmatch = match.group(0)
        return german_chars2[textmatch.encode('utf-8')]
    except KeyError as e:
        print('except')
        return ''


def replace_german_chars(text):
    import re
    clean = re.compile('\\\\xc[0-9]\\\\[a-zA-Z][a-zA-Z0-9][a-zA-Z0-9]')
    return re.sub(clean, myreplace, text)


def replace_double_slashes(text):
    import re
    clean = re.compile('\\\+n')
    return re.sub(clean, '', text)


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = replace_double_slashes(text)
    text = text.strip()  # .replace('\n', '')
    text = text.strip('\\n')
    text = text.replace('\n', ' ')
    text = text.replace('xa', '')
    text = text.replace('xe', '')
    text = text.replace('xb', '')
    return " ".join(text.split())


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


def remove_stop_words(tokenized_text, lang='EN'):
    """remove stop words from text, e.g. the, a, an, me etc"""
    stop_words = {'EN': stop_words_en, 'DE': stop_words_de, 'FR': stop_words_fr}
    text = [w for w in tokenized_text if not w in stop_words[lang]]
    return text


def stemming_words(tokenized_text, lang='EN'):
    # stem's of each word
    snow_stemmer = {'EN': snow_stemmer_en, 'DE': snow_stemmer_de, 'FR': snow_stemmer_fr}
    stem_words = []
    for w in tokenized_text:
        if type(w) == str:
            w = snow_stemmer[lang].stem(w)
        stem_words.append(w)
    return stem_words


def lemmatize_words(tokenized_text, deduplicates=False):
    lemma_list = []
    for w in tokenized_text:
        lemma_list.append(lemmatizer.lemmatize(w))
    if deduplicates:
        lemma_list = list(dict.fromkeys(lemma_list))
    return lemma_list




def remove_single_chars(tokenized_text):
    # stem's of each word
    result = []
    for w in tokenized_text:
        if len(w) > 1:
            result.append(w)
    return result



# Remove numbers
def remove_numbers(text):
    result = nltk.re.sub(r'\d+', '', text)
    return result


def text_preprocessing(text, accented_chars=True, extra_whitespace=True, single_chars_removal=True,
                       lowercase=True, remove_num=True,
                       stop_words=True, stemming=True, convert_lemma=False, lang='EN', ngrams=0):
    """preprocess text with default option set to true for all steps"""
    # print(len(text), 'raw -- \t\t', text)
    if extra_whitespace:  # remove extra whitespaces
        text = remove_whitespace(text)
        # print(len(text), 'whitespace -- \t\t', text)
    if remove_num:
        text = remove_numbers(text)
        # print(len(text), 'remove_num -- \t\t', text)
    if accented_chars:  # remove accented characters
        text = remove_accented_chars(text)
        # print(len(text), 'accented -- \t\t', text)
    if lowercase:  # convert all characters to lowercase
        text = text.lower()
        # print(len(text), 'lowercase -- \t\t', text)
    text = tokenizer.tokenize(text)  # tokenise text
    # print(len(text), 'tokenizer -- \t\t', text)
    if stop_words:  # removes all stop words
        text = remove_stop_words(text, lang=lang)
        # print(len(text), 'stop_words -- \t\t', text)

    if stemming:  # applies Porter Stemmer Algorithm
        text = stemming_words(text, lang=lang)
        # print(len(text), 'stemming word-- \t\t', text)
    if convert_lemma:  # applies Porter Stemmer Algorithm
        text = lemmatize_words(text)
    if single_chars_removal:  # removes single chars
        text = remove_single_chars(text)
    # print(len(text), 'lemmatize -- \t\t', text)
    # text = [i for i in text if len(i) >= 3]
    # print(len(text), 'less than 3 off-- \t\t', text)

    return ' '.join(text)


if __name__ == "__main__":
    print("File one executed when ran directly")
    txt = 'Das Quartäre Eiszeitalter formte die vergleichsweise eintönige Oberflächengeologie Norddeutschlands und des ' \
          'Alpenvorlandes mit ihren Moränen­ablagerungen und sonstigen Begleiterscheinungen großflächiger ' \
          'Vergletscherungen '
    txt = tokenizer.tokenize(txt)
    print(txt)
    d = remove_stop_words(txt, 'DE')
    e = remove_stop_words(txt, 'EN')
    print(d)
    print(e)
    d = stemming_words(d, 'DE')
    e = stemming_words(e, 'EN')
    print(d)
    print(e)