import pandas as pd
import ast

# read in the analysis result per job description
bias_df = pd.read_excel("bias_analysis_results_per_job_description.xlsx")

exact_categories = ['skills_phrases', 'work_env_phrases', 'coding_lang_phrases',
                    'education_phrases', 'experience_phrases', 'advantage_phrases', 'disclaimer_phrases']
similar_categories = ['nlp_additional_phrases']

# create a new dataframe to for mid-calculations
temp_df = pd.DataFrame()

# create two sets of all the phrases in the Excel
temp_df['exact_biased_phrases'] = bias_df.apply(lambda row: [(phrase, category) for category in exact_categories for phrase in (ast.literal_eval(row[category]) if row[category] != '-' else set())], axis=1)
temp_df['similar_biased_phrases'] = bias_df.apply(lambda row: [(phrase, category) for category in similar_categories for phrase in (ast.literal_eval(row[category]) if row[category] != '-' else set())], axis=1)
exact_biased_phrases_set = set()
for phrases in temp_df['exact_biased_phrases']:
    exact_biased_phrases_set.update(phrases)
similar_biased_phrases_set = set()
for phrases in temp_df['similar_biased_phrases']:
    similar_biased_phrases_set.update(phrases)

# Create new columns to indicate whether each row contains each phrase
for phrase, category in exact_biased_phrases_set:
    temp_df[(phrase, category)] = temp_df['exact_biased_phrases'].apply(lambda x: (phrase, category) in x)
for phrase, category in similar_biased_phrases_set:
    temp_df[(phrase, category)] = temp_df['similar_biased_phrases'].apply(lambda x: (phrase, category) in x)

# Count how many rows contain each phrase
exact_phrase_counts = {(phrase, category): temp_df[(phrase, category)].sum() for phrase, category in exact_biased_phrases_set}
similar_phrase_counts = {(phrase, category): temp_df[(phrase, category)].sum() for phrase, category in similar_biased_phrases_set}

# Create a list of dictionaries containing the data
data = [{'phrase': phrase, 'category': category, 'count': count, 'is_original': True} for (phrase, category), count in exact_phrase_counts.items()] \
     + [{'phrase': phrase, 'category': category, 'count': count, 'is_original': False} for (phrase, category), count in similar_phrase_counts.items()]

# Create a DataFrame from the list of dictionaries
results_df = pd.DataFrame(data)

# Calculate the total number of rows in the Excel file
total_rows = len(bias_df)

# Add a new column to the counts_df DataFrame containing the percentage of rows that contain each phrase
results_df['percentage'] = results_df['count'] / total_rows * 100

# Order the result by count descending
results_df_sorted = results_df.sort_values(by='count', ascending=False)

# save the results to a new Excel file
column_order = ['phrase', 'count', 'percentage', 'category', 'is_original']
results_df_sorted.to_excel("bias_analysis_results_per_phrase.xlsx", columns=column_order)
print("Results saved to bias_analysis_results_per_phrase.xlsx")
