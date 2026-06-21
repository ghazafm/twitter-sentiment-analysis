# SocialMediaSentiment

# Sentiment Analysis Tweet Election 2024

This project aims to build a machine learning model that can predict the sentiment of tweets (positive or negative) regarding the 2024 General Election in Indonesia. The model is trained using a dataset from Kaggle that contains tweets on the topic of the 2024 Election along with positive or negative sentiment labels.

## Dataset

The dataset used in this project is sourced from Kaggle:

[Indonesia Presidential Candidates Dataset 2024](https://www.kaggle.com/datasets/jocelyndumlao/indonesia-presidential-candidates-dataset-2024)

This dataset contains Indonesian tweets related to presidential candidates for the 2024 elections, along with positive or negative sentiment labels.

## System Requirements

- Python 3.6 or newer
- Jupyter Notebook
- Library Python:
  - Pandas
  - NumPy
  - Scikit-learn
  - Sastrawi (for stopwords dan stemmer Indonesian language)

## Installation

installation instructions will be added later.

## Usage

To collect a fresh X search sample without browser automation, use the optional
Xquik fetcher:

```bash
export XQUIK_API_KEY="your-api-key"
python scraping/xquik_fetch.py \
  --query "ganjar presiden indonesia 2024 lang:id until:2024-02-29 since:2024-01-01" \
  --output scraping/tweets-data/ganjar.csv \
  --limit 200
```

The script writes the same semicolon-delimited columns used by the existing
notebooks, including tweet IDs, text, engagement counts, language, author, and
tweet URL.

## Contribution

Contribution information will be added later.

## License

This project uses stopwords and stemmer from the Sastrawi library which is under the MIT license.

In addition, the project also uses:
- [Kamus Alay](https://github.com/nasalsabila/kamus-alay) - Dictionary to convert slang to standard Indonesian
- [POS Tagger Bahasa Indonesia dengan Python](https://yudiwbs.wordpress.com/2018/02/20/pos-tagger-bahasa-indonesia-dengan-pytho/) - Tutorial on part-of-speech tagging for Indonesian using Python

## Credit

This project developed by:

- Fauzan Ghaza Madani
- Fatoni Murfid Syaafii

## Related Links

- [Sastrawi](https://pypi.org/project/Sastrawi/) - Library for processing Indonesian text
- [POS Tagger Bahasa Indonesia dengan Python](https://yudiwbs.wordpress.com/2018/02/20/pos-tagger-bahasa-indonesia-dengan-pytho/) - Tutorial on part-of-speech tagging for Indonesian using Python 
