import io
import os
import json
import time
import re
from bs4 import BeautifulSoup as bs

from preprocess_text import text_preprocessing, replace_german_chars, remove_whitespace, replace_double_slashes
from langdetect import detect

start_time = time.time()


# Set collection's root directory
root_dir = 'C:\\Users\\Gmanw\\Desktop\\saltampasis\\v1'

# Output directory where trec files will be exported
output_dir = 'C:\Kyrinio_vol2/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# This list keeps all the patent file names, e.g ['EP-0099910-A4.xml', 'EP-0098920-A1.xml', 'EP-0099001-A3.xml']
# patent_files = []
# This list keeps all the patent file urls, e.g ['/path_to_file/EP-0099910-A4.xml', '/path_to_file/EP-0099001-A2.xml']
patent_file_paths = []

# Population of the lists
for subdir, dirs, files in os.walk(root_dir):
    if files:
        for file in files:
            # patent_files.append(file)
            patent_file_paths.append(subdir + '/' + file)

# This list keeps all patent ids, e.g ['EP-0099910-A4', 'EP-0098920-A1', 'EP-0099001-A3', 'EP-0099001-A2']
# patent_docs = [doc[0:-4] for doc in patent_files]


def deduplicate(a_list):
    tokenized = [' '.join(a.split(' ')[:10]) for a in a_list]
    tokenized_dedupl = list(dict.fromkeys(tokenized))
    results = []
    for index, value in enumerate(tokenized_dedupl):
        if value in tokenized:
            results.append(a_list[tokenized.index(value)])
            continue

    return results
    # return list(dict.fromkeys(tokenized))

def get_trec(bs_object, tag):
    titles = bs_object.find_all(tag)
    processed_titles = []
    for t in titles:
        decoded_title = replace_german_chars(t.text)
        lang = detect(decoded_title).upper()
        if lang not in ['EN', 'DE', 'FR']:
            lang = 'EN'
        processed_titles.append(text_preprocessing(decoded_title, lang=lang))

    # print(len(processed_titles), processed_titles)

    processed_titles = [' '.join(title.split(' ')[:300]) for title in processed_titles]

    # print(len(processed_titles), processed_titles)

    processed_titles = deduplicate(processed_titles)
    # print(len(processed_titles), processed_titles)
    titles_data = ' '.join([title for title in processed_titles])

    if titles_data:
        return '<{}>\n{}\n</{}>\n'.format(titles[0].name.upper(), titles_data, titles[0].name.upper())
    return ''

def get_trec_title(bs_object):
    titles = bs_object.find_all('title')
    processed_titles = []
    for t in titles:
        decoded_title = replace_german_chars(t.text)
        lang = detect(decoded_title).upper()
        if lang not in ['EN', 'DE', 'FR']:
            lang = 'EN'
        processed_titles.append(text_preprocessing(decoded_title, lang=lang))
    processed_titles = deduplicate(processed_titles)
    titles_data = ' '.join([title for title in processed_titles])
    if titles_data:
        return '<TITLE>\n{}\n</TITLE>\n'.format(titles_data)
    return ''


def get_trec_tag(bs_object, tag, preprocess=False, limit=None, **kwargs):
    doc = bs_object.find(tag)
    if doc:
        if not preprocess:
            return '<{}>{}</{}>\n'.format(doc.name.upper(), doc.string, doc.name.upper())
        decoded_text = replace_german_chars(doc.text)
        lang = detect(decoded_text).upper()
        if lang not in ['EN', 'DE', 'FR']:
            lang = 'EN'
        text = text_preprocessing(decoded_text, lang=lang)
        # print(lang, text)
        text = ' '.join(text.split(' ')[:300])
        # print('\n',text)

        return '<{}>\n{}\n</{}>\n'.format(doc.name.upper(), text, doc.name.upper())
    return ''


def close_doc():
    return '</TEXT>\n</DOC>'


def get_classification(bs_object):
    cls = bs_object.find('classifications-ipcr')
    return '<{}>\n{}\n</{}>\n'.format(cls.name.upper()," ".join(cls.text.replace("'", "").lower().strip().split()), cls.name.upper())

#patent_file_paths = ['D:\Collection_Assignment_Part2_final/16/EP-0295736.txt', 'D:\Collection_Assignment_Part2_final/17/EP-0316749.txt', 'D:\Collection_Assignment_Part2_final/19/EP-0336409.txt']

with io.open('D:\error_log_2.csv', "w+", encoding="utf-8") as error_log:
    for count, file in enumerate(patent_file_paths):
        try:
            xml = open(file, 'r', encoding='utf-8').read()
            root = bs(xml, 'html.parser')
        except Exception:
            continue

        # xml = open('D:\Collection_Assignment_Part2_final/14/EP-0259842.txt', 'r', encoding='utf-8').read()
        # root = bs(xml, 'html.parser')
        print(count, file)
        # print(get_trec(root, 'claims'))
        # if count > 5:
        #     exit()
        # input('www')
        # continue

        filename = file.split('/')[-1].split('.')[0]
        with io.open(os.path.join(output_dir, '{}.txt'.format(filename)), "w+", encoding="utf-8") as f:
            f.write('<DOC>\n')
            try:
                f.write(get_trec_tag(root, 'docno'))
            except Exception as e:
                error_log.write('{};docno;{}\n'.format(file, e))
            f.write('<TEXT>\n')
            try:
                f.write(get_trec_title(root))
            except Exception as e:
                error_log.write('{};title;{}\n'.format(file, e))
            try:
                f.write(get_classification(root))
            except Exception as e:
                error_log.write('{};classification;{}\n'.format(file, e))
            try:
                f.write(get_trec(root, 'abstract'))#(get_trec_tag(root, 'abstract', preprocess=True))
            except Exception as e:
                error_log.write('{};abstract;{}\n'.format(file, e))
            try:
                f.write(get_trec(root, 'description'))#(get_trec_tag(root, 'description', preprocess=True))
            except Exception as e:
                error_log.write('{};description;{}\n'.format(file, e))
            try:
                f.write(get_trec(root, 'claims'))#(get_trec_tag(root, 'claims', preprocess=True))
            except Exception as e:
                error_log.write('{};claims;{}\n'.format(file, e))
            try:
                f.write(close_doc())
            except Exception as e:
                error_log.write('{};close_doc;{}\n'.format(file, e))
            f.close()
        #break
    error_log.close()

print("Executed in {} minutes.".format(round((time.time() - start_time) / 60), 2))