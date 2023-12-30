# bring new data (only accept csv )

# transform data

# split to train test

# train model on train data

# fine tune

# compare new model result with old model result

# if better save new model

# monitoring and logging


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import logging
from soil_fertility.utils import retreive_base_path, load_object, save_object
import os
from soil_fertility.components.utils.model_utils import fix_columns_name


class TrainPipeline:
    def __init__(self):
        base_path = retreive_base_path()
        artifacts_path = os.path.join(base_path, "artifacts")
        self.preprocessors_path = os.path.join(artifacts_path, "preprocessors")

    def train(self, csv_file_path, old_model_path):
        try:
            all_processor_path = os.path.join(
                self.preprocessors_path, "all_features_pipeline.pkl"
            )
            preprocessor_path = os.path.join(
                self.preprocessors_path, "preprocessor_first.pkl"
            )

            data = pd.read_csv(csv_file_path)
            preprocessor = load_object(preprocessor_path)
            all_processor = load_object(all_processor_path)
            model = load_object(old_model_path)
            fixed_data = all_processor.transform(data)
            scaled_data = preprocessor.transform(fixed_data)
            data = fix_columns_name(scaled_data)

            # Splitting Data
            X_train, X_test, y_train, y_test = train_test_split(
                data.drop("Fertility", axis=1), data["Fertility"], test_size=0.2
            )
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()

            # Model Training
            model.fit(X_train, y_train)

            # Model Comparison
            old_model = load_object(old_model_path)
            new_score = accuracy_score(y_test, model.predict(X_test))
            old_score = accuracy_score(y_test, old_model.predict(X_test))

            # Model Updating
            if new_score > old_score:
                logging.info(
                    f"New model score: {new_score} is higher than old model score: {old_score}, new model will be saved"
                )
                print(
                    f"New model score: {new_score} is higher than old model score: {old_score}, new model will be saved"
                )
                save_object(obj=model, file_path=old_model_path)
            else:
                print(
                    f"Old model score: {old_score} is higher than new model score: {new_score}, old model will be kept"
                )
                logging.info(
                    f"Old model score: {old_score} is higher than new model score: {new_score}, old model will be kept"
                )

        except Exception as e:
            logging.error(f"Exception occurred: {e}")
            raise e
