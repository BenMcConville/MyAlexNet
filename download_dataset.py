from dep.SetCreator import *

# Create instance of training/validation loader
train_loader, valid_loader = create_training_validation_set('./data', download=True)

# Create instance of test loader
test_loader = create_testing_set('./data', download=True)
