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
df = pd.read_excel("job_descriptions_linkedin.xlsx")

# read in the biased phrases excel
bias_df = pd.read_excel("biased_phrases.xlsx")

# convert to lowercase
bias_df['phrase'] = bias_df['phrase'].str.lower()

# create a set of all biased phrases (exact matches)
exact_biased_phrases = set(bias_df['phrase'].tolist())


# define a function to check if a description contains any biased phrases (exact or similar matches)
def get_job_desc_statistics_for_biased_phrases(description):
    description = description.lower()
    desc_tokens = [token.text for token in nlp(description.lower())]
    matches = {'skills': [], 'work_env': [], 'nlp_phrases': [], 'education': [], 'coding_lang': [], 'experience': [], 'advantage': [], 'disclaimer': [], 'job_desc_line_count': 0}

    for phrase in exact_biased_phrases:
        if phrase.lower() in description:
            category = bias_df[bias_df['phrase'] == phrase]['category'].values[0]  # very inefficient
            matches[category].append(phrase)
        else:
            closest_match = difflib.get_close_matches(phrase, desc_tokens, n=1, cutoff=0.8)
            if closest_match and closest_match[0] not in STOPWORDS:
                matches['nlp_phrases'].append(closest_match[0])

    matches['job_desc_line_count'] = len(description.split('\n'))
    return matches


def get_statistics_for_biased_phrases():
    # create a new column indicating which biased phrases are present
    df['matches'] = df['description'].apply(get_job_desc_statistics_for_biased_phrases)

    # create a new dataframe to store the results
    results_df = pd.DataFrame(columns=['description'])

    # iterate over the job descriptions to calculate the percentage of biased phrases present in each
    for index, row in df.iterrows():
        description = row['description']

        # get matches per category
        skills_phrases = set(row['matches']['skills'])
        work_env_phrases = set(row['matches']['work_env'])
        coding_lang_phrases = set(row['matches']['coding_lang'])
        education_phrases = set(row['matches']['education'])
        experience_phrases = set(row['matches']['experience'])
        advantage_phrases = set(row['matches']['advantage'])
        disclaimer_phrases = set(row['matches']['disclaimer'])
        nlp_phrases = set(row['matches']['nlp_phrases'])

        # we want the additional nlp phrases to be different from any exact match
        nlp_phrases -= skills_phrases
        nlp_phrases -= work_env_phrases
        nlp_phrases -= coding_lang_phrases
        nlp_phrases -= education_phrases
        nlp_phrases -= experience_phrases
        nlp_phrases -= advantage_phrases
        nlp_phrases -= disclaimer_phrases

        # calculate matches count
        skills_count = len(skills_phrases)
        work_env_count = len(work_env_phrases)
        coding_lang_count = len(coding_lang_phrases)
        education_count = len(education_phrases)
        experience_count = len(experience_phrases)
        advantage_count = len(advantage_phrases)
        disclaimer_count = len(disclaimer_phrases)
        nlp_count = len(nlp_phrases)
        total_matches_count = skills_count + work_env_count + coding_lang_count + education_count + experience_count + nlp_count

        # add the job description results to dataframe
        results_df = results_df.append({'job_description': description, 'job_desc_line_count': row['matches']['job_desc_line_count'],
                                        'skills_phrases': skills_phrases if skills_count > 0 else "-", 'skills_count': skills_count,
                                        'work_env_phrases': work_env_phrases if work_env_count > 0 else "-", 'work_env_count': work_env_count,
                                        'coding_lang_phrases': coding_lang_phrases if coding_lang_count > 0 else "-", "coding_lang_count": coding_lang_count,
                                        'education_phrases': education_phrases if education_count > 0 else "-", 'education_count': education_count,
                                        'experience_phrases': experience_phrases if experience_count > 0 else "-", 'experience_count': experience_count,
                                        'advantage_phrases': advantage_phrases if advantage_count > 0 else "-", 'advantage_count': advantage_count,
                                        'disclaimer_phrases': disclaimer_phrases if disclaimer_count > 0 else "-", 'disclaimer_count': disclaimer_count,
                                        'nlp_additional_phrases': nlp_phrases if nlp_count > 0 else "-", 'nlp_additional_count': nlp_count, 'total_matches_count': total_matches_count}, ignore_index=True)

    # save the results to a new Excel file
    column_order = ['description', 'job_desc_line_count', 'skills_phrases', 'skills_count', 'work_env_phrases', 'work_env_count', 'coding_lang_phrases',
                    'coding_lang_count', 'education_phrases', 'education_count', 'experience_phrases', 'experience_count', 'advantage_phrases',
                    'advantage_count', 'disclaimer_phrases', 'disclaimer_count', 'nlp_additional_phrases', 'nlp_additional_count', 'total_count']
    results_df.to_excel("bias_analysis_results_per_job_description.xlsx", columns=column_order)
    print("Results saved to bias_analysis_results_per_job_description.xlsx")


if __name__ == "__main__":
    get_statistics_for_biased_phrases()
