import pandas as pd
import spacy
import difflib
import nltk
from nltk.corpus import stopwords

try:
    nlp = spacy.load("en_core_web_lg")
except:
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

STOPWORDS = {}
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words('english'))

# read in the job descriptions excel
df = pd.read_excel("job_descriptions_2.xlsx")

# read in the biased phrases excel
bias_df = pd.read_excel("biased_keywords.xlsx")

# create a set of all biased phrases (exact matches)
exact_biased_phrases = set(bias_df['phrase'].tolist())

# create a dictionary of all biased phrases (similar matches) and their corresponding categories
similar_biased_phrases = {}
for index, row in bias_df.iterrows():
    category = row['category']
    phrase = row['phrase']
    if phrase not in exact_biased_phrases:
        if category not in similar_biased_phrases:
            similar_biased_phrases[category] = [phrase]
        else:
            similar_biased_phrases[category].append(phrase)


# define a function to check if a description contains any biased phrases (exact or similar matches)
def check_for_bias(description):
    doc = nlp(description.lower())
    matches = []
    for token in doc:
        token_matches = {'phrase': token.text, 'category': None, 'original': True}
        if token.text in exact_biased_phrases:
            token_matches['category'] = bias_df[bias_df['phrase'] == token.text]['category'].values[0]
            matches.append(token_matches)
        else:
            for category in similar_biased_phrases:
                closest_match = difflib.get_close_matches(token.text, similar_biased_phrases[category], n=1, cutoff=0.3)
                if closest_match:
                    token_matches['category'] = category
                    token_matches['phrase'] = closest_match[0]
                    token_matches['original'] = False
                    matches.append(token_matches)
    return matches


# create a new column in the job descriptions excel indicating which biased phrases are present
df['biased_phrases'] = df['description'].apply(check_for_bias)

# create a dictionary of all biased phrases and their corresponding categories
all_biased_phrases = {}
for index, row in bias_df.iterrows():
    category = row['category']
    phrase = row['phrase']
    all_biased_phrases[phrase] = category

# calculate the percentage of job descriptions that contain each biased phrase
counts = {}
for index, row in df.iterrows():
    for phrase in row['biased_phrases']:
        if phrase['phrase'] not in counts:
            counts[phrase['phrase']] = {'count': 1, 'percent': 0, 'category': phrase['category'], 'original': phrase['original']}
        else:
            counts[phrase['phrase']]['count'] += 1
for phrase in counts:
    counts[phrase]['percent'] = counts[phrase]['count'] / len(df) * 100

# create a new dataframe to store the results
results_df = pd.DataFrame.from_dict(counts, orient='index', columns=['count', 'percent', 'category', 'original'])

# save the results to a new Excel file
results_df.to_excel("bias_analysis_results.xlsx")

print("Results saved to bias_analysis_results.xlsx")
