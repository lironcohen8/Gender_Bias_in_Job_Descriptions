import pandas as pd

# Load the job descriptions from the Excel file
df = pd.read_excel('job_descriptions_linkedin.xlsx')

# Create a new DataFrame to store the sentences
sentences_df = pd.DataFrame(columns=['Sentence'])

# Iterate over the job descriptions
for description in df['description']:
    try:
        # Split the description into sentences
        sentences = description.replace('\n', '.').split('.')
        # Iterate over the sentences
        for sentence in sentences:
            # Check if the sentence is not too short
            sentence = sentence.strip()
            if len(sentence.split()) > 4:
                # Add the sentence to the DataFrame
                sentences_df = sentences_df.append({'Sentence': '"'+sentence+'"\n'}, ignore_index=True)
    except:
        print(f"Exception occurred.")

# Save the sentences to a new Excel file
sentences_df.to_excel('job_descriptions_sentences.xlsx', index=False)
