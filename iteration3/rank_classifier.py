import pandas as pd
import numpy as np
import gdown
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import pickle
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

url = 'https://drive.google.com/uc?id=1BRYal9JVBF5DNji0Bh2Vjtf5MEpwv8Tx'
output = 'new_data.csv'
model_filename = 'rank_prediction_model.pkl'

def download_file(url, output):
    try:
        gdown.download(url, output, quiet=False)
    except Exception as e:
        print("Error downloading file:", str(e))
        # Handle the error appropriately (e.g., retry, exit, etc.)


def train_classifier(X_train_tfidf, y_train):
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    return clf


def evaluate_classifier(clf, X_test_tfidf, y_test):
    predicted = clf.predict(X_test_tfidf)
    accuracy = metrics.accuracy_score(y_test, predicted)
    print("Accuracy:", accuracy)
    
    return predicted


def save_model(clf, filename):
    with open(filename, 'wb') as file:
        pickle.dump(clf, file)


def load_model(filename):
    with open(filename, 'rb') as file:
        clf = pickle.load(file)
    return clf


def create_heatmap(cm_normalized, labels):
    cm_normalized_df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
    
    # Create annotations for the heatmap
    annot = np.empty_like(cm_normalized).astype(str)
    nrows, ncols = cm_normalized.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm_normalized[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = f'{c * 100:.1f}%'

    # Plot the heatmap
    sns.heatmap(cm_normalized_df, annot=annot, fmt='', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def predict_sentence_rank(sentence, clf):
    # Preprocess the input sentence
    sentence_counts = count_vect.transform([sentence])
    sentence_tfidf = tfidf_transformer.transform(sentence_counts)

    # Predict the rank
    predicted_rank = clf.predict(sentence_tfidf)

    return predicted_rank[0]

def plot_class_distribution(y):
    class_counts = y.value_counts()
    class_labels = class_counts.index

    plt.figure()
    ax = sns.barplot(x=class_labels, y=class_counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    
    # Add numbers on the bars
    for i, count in enumerate(class_counts):
        ax.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
    
    plt.show()

def plot_precision_recall_curve(clf, X_test_tfidf, y_test):
    precision = dict()
    recall = dict()
    thresholds = dict()

    # Calculate precision and recall for each class
    for i, class_label in enumerate(clf.classes_):
        class_index = np.where(clf.classes_ == class_label)[0][0]
        y_test_binary = np.where(y_test == class_label, 1, 0)
        probabilities = clf.predict_proba(X_test_tfidf)[:, class_index]
        precision[class_label], recall[class_label], thresholds[class_label] = precision_recall_curve(
            y_test_binary, probabilities
        )

        # Plot the precision-recall curve for the class
        plt.step(recall[class_label], precision[class_label], where='post', label=f'Class {class_label}')

    # Plot the mean precision-recall curve
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

def plot_roc_curve(clf, X_test_tfidf, y_test):
    predicted = clf.predict(X_test_tfidf)
    accuracy = metrics.accuracy_score(y_test, predicted)
    print("Accuracy:", accuracy)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure()  # Create a single figure to hold all ROC curves

    # Calculate FPR, TPR, and AUC for each class
    for i, class_label in enumerate(clf.classes_):
        class_index = np.where(clf.classes_ == class_label)[0][0]
        y_test_binary = np.where(y_test == class_label, 1, 0)
        probabilities = clf.predict_proba(X_test_tfidf)[:, class_index]

        # Calculate FPR, TPR, and AUC for ROC curve
        fpr[class_label], tpr[class_label], _ = roc_curve(y_test_binary, probabilities)
        roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])

        # Plot ROC curve on the same figure
        plt.plot(fpr[class_label], tpr[class_label], label=f'Class {class_label} (AUC = {roc_auc[class_label]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def train_model_and_create_heatmap():
    # Download the file
    download_file(url, output)

    # Read the data
    df = pd.read_csv('new_data.csv')
    df.head()

    # Bar Plot of Class Distribution
    plot_class_distribution(df['rank'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df['Sentence'], df['rank'], random_state=0)

    # Vectorize and transform the data
    X_train_counts = count_vect.fit_transform(X_train)

    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Train the classifier
    clf = train_classifier(X_train_tfidf, y_train)

    # Transform test data
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    # Random oversampling
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X_train_tfidf, y_train)

    # Train the classifier with resampled data
    clf_resampled = train_classifier(X_resampled, y_resampled)

    # Save the trained model
    save_model(clf_resampled, model_filename)

    # Evaluate the classifier
    predicted_resampled = evaluate_classifier(clf_resampled, X_test_tfidf, y_test)

    # Plot Precision-Recall curve
    plot_precision_recall_curve(clf_resampled, X_test_tfidf, y_test)

    # Create the confusion matrix
    cm_resampled = confusion_matrix(y_test, predicted_resampled)
    cm_resampled_normalized = cm_resampled.astype('float') / cm_resampled.sum(axis=1)[:, np.newaxis]
    labels = np.unique(y_test)

    # Create the heatmap
    create_heatmap(cm_resampled_normalized, labels)

    # Plot ROC curve
    plot_roc_curve(clf_resampled, X_test_tfidf, y_test)

    return clf_resampled


def main():
    df = pd.read_excel('job_descriptions_linkedin.xlsx')
    description_ranks = []
    loaded_clf = train_model_and_create_heatmap()
    # loaded_clf = load_model(model_filename)

    try:
        for description in df['description']:
            sentences = description.replace('\n', '.').split('.')
            sentence_ranks = []

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) > 4:
                    sentence_rank = predict_sentence_rank(sentence, loaded_clf)
                    sentence_ranks.append(sentence_rank)
            description_rank = sum(sentence_ranks) / len(sentence_ranks)
            description_rank = round(description_rank, 3)
            description_ranks.append(description_rank)

        df['description_rank'] = description_ranks
        df.to_excel('job_descriptions_ranks.xlsx', index=False)

    except:
        print("Exception occurred.")


if __name__ == '__main__':
    main()