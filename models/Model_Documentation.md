**Training & Evaluating Models:**

Directory Structure:

model_definitions
Definitions of Models
model_evaluators
Code to train & evaluate models
model_utils

Code to perform the act of training as well as other generic reusable utilities

Adding a new model:
1. in model_definitions define the new model. You may copy/paste another one and modify it (remember to change the name!)
2. in model_evaluators, copy & modify the code to train and evaluate the models
3. in model_utils.train_utils, decide which train method to use. Or create your own. Either create your own method, or commit changes to all models. Be aware you are changing code used by all models, so only commit changes here if they improve performance. Ideally quickly test one of the other models (set epochs to 1, just make sure they train without breaking anything) 