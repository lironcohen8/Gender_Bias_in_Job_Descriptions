import pandas as pd
import spacy
import difflib
spacy.cli.download("en_core_web_lg")

# load the spaCy model
nlp = spacy.load("en_core_web_lg")

# read in the job descriptions excel
df = pd.read_excel("job_descriptions.xlsx")

# read in the biased phrases excel
bias_df = pd.read_excel("biased_keywords.xlsx")

# convert to lowercase
bias_df['phrase'] = bias_df['phrase'].str.lower()

# create a set of all biased phrases (exact matches)
exact_biased_phrases = set(bias_df['phrase'].tolist())


# define a function to check if a description contains any biased phrases (exact or similar matches)
def check_for_bias(description):
    print("here")
    description = description.lower()
    desc_tokens = [token.text for token in nlp(description.lower())]
    matches = {'Skills': [], 'work environment': [], 'nlp_phrases': [], 'education': [], 'coding lang': []}
    for phrase in exact_biased_phrases:
        if phrase.lower() in description:
            category = bias_df[bias_df['phrase'] == phrase]['category'].values[0]  # very inefficient
            matches[category].append(phrase)
        else:
            closest_match = difflib.get_close_matches(phrase, desc_tokens, n=1, cutoff=0.8)
            if closest_match:
                matches['nlp_phrases'].append(closest_match[0])
    return matches


def main():
    # create a new column indicating which biased phrases are present
    df['matches'] = df['description'].apply(check_for_bias)

    # create a new dataframe to store the results
    results_df = pd.DataFrame(columns=['description'])

    # iterate over the job descriptions to calculate the percentage of biased phrases present in each
    for index, row in df.iterrows():
        description = row['description']

        skills_biased_phrases = set(row['matches']['Skills'])
        work_env_biased_phrases = set(row['matches']['work environment'])
        coding_lang_biased_phrases = set(row['matches']['coding lang'])
        education_biased_phrases = set(row['matches']['education'])
        nlp_biased_phrases = set(row['matches']['nlp_phrases'])

        nlp_biased_phrases -= skills_biased_phrases
        nlp_biased_phrases -= work_env_biased_phrases
        nlp_biased_phrases -= coding_lang_biased_phrases
        nlp_biased_phrases -= education_biased_phrases

        skills_count = len(skills_biased_phrases)
        work_env_count = len(work_env_biased_phrases)
        coding_lang_count = len(coding_lang_biased_phrases)
        education_count = len(education_biased_phrases)
        nlp_count = len(nlp_biased_phrases)
        total_count = skills_count + work_env_count + coding_lang_count + education_count + nlp_count

        results_df = results_df.append({'description': description, 'skills_biased_phrases': skills_biased_phrases,
                                        'skills_count': skills_count, 'work_env_biased_phrases': work_env_biased_phrases,
                                        'work_env_count': work_env_count, 'coding_lang_biased_phrases': coding_lang_biased_phrases,
                                        "coding_lang_count": coding_lang_count, 'education_biased_phrases': education_biased_phrases,
                                        'education_count': education_count, 'nlp_biased_phrases': nlp_biased_phrases,
                                        'nlp_count': nlp_count, 'total_count': total_count}, ignore_index=True)

    # save the results to a new Excel file
    results_df.to_excel("bias_analysis_results.xlsx")


if __name__ == "__main__":
    main()
