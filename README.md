# CurriculumClassifier
Classifier extending scikit-learn, that uses curriculum learning.

## Steps
- Funkcija ki z RandomForestClassifier razbije senzitivne skupine
- Classifier (ki extenda BaseEstimator, itd…), ki ima kot input tip classifierja (Ridge, etc - mora support partial fit) in kak naj se curriculum learning izvaja (tezje -> lazje, lazje -> tezje)
- Potestiras s fairleran adult dataset (primerjas s nekimi drugimi npr Ridge) 