# Imports
import os
import json
import numpy as np

# Result File Structure
#   result
#     - model_name
#       - dataset_name
#        - fold_#
#          - accuracy_#.png
#          - loss_#.png
#          - results_#.json
#   torch_code
#     - result_analysis.py (this file)

class Result:

    def __init__(self, model, dataset, num_folds):
        self.model = model
        self.dataset = dataset
        self.num_folds = num_folds

        self.kfold_results = []

    def get_results(self):
        result = {
            "Model": self.model,
            "Dataset": self.dataset,
            "Folds": self.num_folds,
            "Results": self.kfold_results
        }

        return result

    # Run add_results
    def add_results(self, num_epochs):
        file_path = os.path.join(
            '..',
            'results', 
            self.model, 
            self.dataset, 
            f'epochs_{num_epochs}'
        )

        cum_results = {
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F-1_Score": [],
            "Best_Epoch": []
        }

        fold_count = 0

        for fold in range(self.num_folds):
            file_name = os.path.join(
                file_path,
                f'fold_{fold}',
                f'results_{fold}.json'
            )

            if os.path.exists(file_name):

                fold_count += 1

                print(file_name + " exists")

                with open(file_name) as f:
                    results = json.load(f)

                cum_results["Accuracy"].append(float(results["Accuracy"].rstrip('%')))
                cum_results["Precision"].append(float(results["Precision"].rstrip('%')))
                cum_results["Recall"].append(float(results["Recall"].rstrip('%')))
                cum_results["F-1_Score"].append(float(results["F-1_Score"].rstrip('%')))
                cum_results["Best_Epoch"].append(int(results["Best_Epoch"]["epoch"]))

        

        results = {
            "Epochs": num_epochs,
            "Accuracy": round(np.mean(cum_results["Accuracy"]), 3),
            "Precision": round(np.mean(cum_results["Precision"]), 3),
            "Recall": round(np.mean(cum_results["Recall"]), 3),
            "F-1_Score": round(np.mean(cum_results["F-1_Score"]), 3),
            "Best_Epoch": int(np.ceil(np.mean(cum_results["Best_Epoch"]))),
            "Folds_Averaged": fold_count,
        }

        self.kfold_results.append(results)

def main():
    results = {}

    models = ["resnet18", "resnet50", "vit_b_16", "vit_b_32"]
    datasets = ["ucmerced_landuse", "cifar10"]
    num_folds = 5

    for dataset in datasets:
        dataset_results = []
        for model in models:
            res = Result(model, dataset, num_folds)
            res.add_results(num_epochs=50)

            dataset_results.append(res.get_results())

        results[dataset] = dataset_results

    print(results)

    file_path = os.path.join('..', 'results', 'combined_results.json')
    with open(file_path, 'w') as f:
        json.dump(results, f)

    print("Results Saved to: " + file_path)



if __name__ == "__main__":
    main()