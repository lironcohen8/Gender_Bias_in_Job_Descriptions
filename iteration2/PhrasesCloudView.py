import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load data from Excel file
df = pd.read_excel('bias_analysis_results_per_phrase_for_wordcloud.xlsx')

# Create a dictionary with phrases as keys and counts as values
data = dict(zip(df['phrase'], df['count']))

# Create and generate a word cloud image
wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(data)

# Display the generated image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
