import sys

from dataset import Dataset
from classificationmodel import ClassificationModel as cm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# main function
if __name__ == "__main__":

    # checking number of args
    if len(sys.argv) < 3:
        print("Usage: python main.py [-create dataset_binary] [-export export_path] [-cv folds_num] [-import import_path] [-tune import_path]")
        exit(1)

    # examining options
    _create = False
    _export = False
    _import = False
    _tune = False
    _cv = False
     
    model = None
    dataset_binary_path = ''
    export_path = ''
    import_path = ''
    folds = 5

    for i in range(0, len(sys.argv)):
        if sys.argv[i] == '-create':
            _create = True
            dataset_binary_path = sys.argv[i+1]
            i += 1
        if sys.argv[i] == '-import':
            _import = True
            import_path = sys.argv[i+1]
            i += 1
        if sys.argv[i] == '-export':
            _export = True
            export_path = sys.argv[i+1]
            i += 1
        if sys.argv[i] == '-cv':
            _cv = True
            folds = int(sys.argv[i+1])
            i += 1
        if sys.argv[i] == '-tune':
            _tune = True
            import_path = sys.argv[i+1]
            i += 1

    # if create new model...
    if _create:
        try:
            # reading dataset and extracting features
            dset = Dataset.deserialize(dataset_binary_path)
            print("Dataset loaded from " + dataset_binary_path)
            # building classification model object
            model = cm(dset)
            print("Model created")
            # training model on training set
            model.train()
            print("Model trained on training set")
            # testing model on test set
            model.test('test_report.csv')
            print("Model evaluated on test set")
        except NameError:
            print('Error while creating model')
            
    # if import model...
    if _import:
        try:
            model = cm.deserialize(import_path)
            print("Model imported from " + import_path)
        except NameError:
            print('Error while importing model')

    # if tune model...
    if _tune:
        try:
            model = cm.deserialize(import_path)
            print("Model imported from " + import_path)
            print("Now tuning...")
            model.tune()
        except NameError:
            print('Error while importing model')

    # if export model...
    if _create:
        try:
            model.serialize(export_path)
            print("Model exported to " + export_path)
        except NameError:
            print('Error while exporting model')
            
    # if cross validation model...
    if _cv:
        try:
            model.cross_validation(folds)
        except NameError:
            print('Error while doing cross validation')