#Assignment 3

## Author
* **Ioannis Manolas**
---

## Assignment Specification

This final part involves building a complete text preprocessing and TREC-format conversion pipeline for multi-lingual patent documents. The tasks include:
* **Advanced Text Preprocessing (`preprocess_text.py`)**: Multi-lingual support (English, French, German) for handling stopwords, Snowball stemming, lemmatization, German character encoding normalization, unidecode accent removal, whitespace cleaning, and regex-based tokenization[cite: 7].
* **TREC Format Conversion (`2.py`)**: Recursively traversing a patent dataset directory, parsing XML documents using BeautifulSoup, detecting document languages (`langdetect`), deduplicating text fragments, handling extraction errors, and exporting structured TREC-formatted `.txt` files[cite: 8, 9].

---

## Solution Overview

The solution consists of two main modules working together to clean and format the raw patent collection into a standardized TREC corpus[cite: 7, 8, 9].

### 1. Preprocessing Pipeline (`preprocess_text.py`)
* **Character Normalization:** Corrects German special characters and encoding artifacts, and strips accented characters using `unidecode`[cite: 7].
* **Tokenization & Stopwords:** Tokenizes text using regex (`\w+`) and filters language-specific stopwords for English, French, and German using `nltk`[cite: 7].
* **Stemming & Lemmatization:** Applies language-specific `SnowballStemmer` and `WordNetLemmatizer`[cite: 7].

### 2. TREC Document Builder (`2.py`)
* **Directory Traversal:** Recursively scans the root directory to locate all patent files[cite: 8, 9].
* **Language Detection:** Dynamically detects text language (`EN`, `DE`, `FR`) using `langdetect` to apply appropriate preprocessing rules[cite: 8, 9].
* **Deduplication:** Cleans and deduplicates overlapping text sections to ensure high corpus quality[cite: 8, 9].
* **Error Logging:** Captures and logs any parsing or extraction exceptions into an error CSV file (`error_log_2.csv`)[cite: 9].
* **TREC Output Generation:** Exports each processed patent into a standardized TREC format containing `<DOC>`, `<TEXT>`, `<TITLE>`, `<CLASSIFICATIONS-IPCR>`, `<ABSTRACT>`, `<DESCRIPTION>`, and `<CLAIMS>` tags[cite: 9].

### Python Libraries Used:
* **`nltk`**: For tokenization, stopwords, stemming, and lemmatization[cite: 7].
* **`unidecode`**: For removing accented characters[cite: 7].
* **`beautifulsoup4`**: For parsing XML/HTML patent structures[cite: 7, 8, 9].
* **`langdetect`**: For identifying document and section languages[cite: 8, 9].

---

## Important Note

To execute the scripts successfully, make sure to update the local root directory and output paths in `2.py` to match your environment[cite: 8, 9]:
```python
root_dir = 'path/to/your/patents'
output_dir = 'path/to/your/output'
