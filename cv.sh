echo "===== path-similarity ====="
python main.py -import ./models/model.words_kartelj_svm-path_similarity.pkl -cv 5
echo "===== lch-similarity ====="
python main.py -import ./models/model.words_kartelj_svm-lhc.pkl -cv 5
echo "===== wu palmer-similarity ====="
python main.py -import ./models/model.words_kartelj_svm-wup.pkl -cv 5
