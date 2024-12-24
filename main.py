from fairlearn.datasets import fetch_adult

from curriculum_classifier.utils.utils import get_sensitive_groups

if __name__ == '__main__':
    df = fetch_adult(as_frame=True)['data']
    print(df['race'].unique())