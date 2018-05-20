# Word recognizer
## Requirements:
1. Tensorflow 
2. python_speech_features
3. sklearn
4. scipy

## Pipeline:
1. Run utils/cutting.py providing data directory
2. Run utils/train_test_split providing cut_data directory
3. To train:
..+ authors network run train.py providing cut_data/train and cut_data/test
..+ comparative models run comp_train.py providing cut_data/train and cut_data/test, change network type in the code for your convenience
