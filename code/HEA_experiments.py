import os
import pandas as pd
import numpy as np
import random
import itertools
import json
import yaml
from ml_model.classifier import init_classifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RepeatedKFold, ShuffleSplit, GridSearchCV, LeaveOneOut
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser
from collections import Counter
from EvidentialClassifier.similarity_combination_element_lib import HybridClassifier, ExpertParser

def make_dirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def summarize_evaluation(df_evaluation, final_decisions, y_samples):
    y_preds, m_High, m_Low, m_Unk = [], [], [], []
    for final_decision in final_decisions:
        m_High.append(final_decision[frozenset({"High"})])
        m_Low.append(final_decision[frozenset({"Low"})])
        m_Unk.append(final_decision[frozenset({"High", "Low"})])
        y_preds.append("High" if final_decision[frozenset({"High"})]>final_decision[frozenset({"Low"})] else "Low")
    df_evaluation["m_High"] = m_High
    df_evaluation["m_Low"] = m_Low
    df_evaluation["m_Unk"] = m_Unk
    df_evaluation["pred_Label"] = y_preds
    df_evaluation["sampling_Label"] = y_samples

    acc_score = accuracy_score(df_evaluation["Label"].values, df_evaluation["pred_Label"].values)
    acc_sigificant_score = accuracy_score(df_evaluation[df_evaluation["m_Unk"]!=1]["Label"].values, df_evaluation[df_evaluation["m_Unk"]!=1]["pred_Label"].values)
    acc_score_sample = accuracy_score(df_evaluation["Label"].values, df_evaluation["sampling_Label"].values)
    return df_evaluation, acc_score, acc_sigificant_score, acc_score_sample

def split_dataset(df_data, df_candidates, n_splits=10, n_repeats=3, keep_balance=False, cv_config="random"):
    mode = cv_config["mode"]
    if mode == "random":
        if isinstance(n_splits, int):
            kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=23)
        else:
            kf = ShuffleSplit(n_splits=n_repeats, train_size=n_splits, random_state=1)
        high_index = df_data[df_data["Label"]=="High"].index.values
        low_index = df_data[df_data["Label"]=="Low"].index.values
        if keep_balance:
            for (train_high, test_high), (train_low, test_low) in zip(kf.split(high_index), kf.split(low_index)):
                yield np.concatenate((high_index[train_high], low_index[train_low]), axis=None), \
                    np.concatenate((high_index[test_high], low_index[test_low]), axis=None)
        else:
            for train, test in kf.split(df_data.index.values):
                yield train, test
    elif mode == "element":
        for element in cv_config["params"]:
            # train = df_data[df_data[element]!=1].index.values
            # test = df_data[df_data[element]==1].index.values
            # yield train, test
            train, test = [], []
            for i, (ith, row) in enumerate(df_data.iterrows()):
                if row[element] == 1:
                    test.append(i)
                else:
                    train.append(i)
            yield train, test
    elif mode == "loo":
        loo = LeaveOneOut()
        for train, test in loo.split(df_data.index.values):
            yield train, test
    elif mode == "recommendation":
        yield df_data.index.values, df_candidates.index.values
    else:
        query = cv_config["query"]
        # df_tmp = df_data[df_data[mode]==1]
        df_tmp = df_data.query(query)
        for i in range(n_repeats):
            df_init = df_tmp.sample(n=int(len(df_tmp)*n_splits), random_state=i)
            train, test = [], []
            for ith, row in df_data.iterrows():
                if ith in df_init.index.values:
                    train.append(ith)
                else:
                    test.append(ith)
            yield train, test

def feature_selection(model, X, y, k=8):
    selector = SelectKBest(k=k)
    X_transformed = selector.fit_transform(X, y)
    mask_features = selector.get_support()
    return X_transformed, mask_features

def parse_arguments():
	parser = ArgumentParser(description="DST with ChatGPT")
	parser.add_argument("--config", metavar="FILE", help="Config file.")
    parser.add_argument("--spark", default=False, help="Enable spark mode.")
    return parser.parse_args()

def main(config, spark):
    # Load data
    df_data = pd.read_csv(config["input_path"]["calculations"], index_col=0)
    df_candidates = pd.read_csv(config["input_path"]["candidates"], index_col=0)
    df_data = df_data[df_data["length"]==np.max(config["input_path"]["n_element"])]
    df_candidates = df_candidates[df_candidates["length"]==np.max(config["input_path"]["n_element"])]
    if config["input_path"]["mask_matrix"] == "None":
        df_mask_matrix = None
    else:
        df_mask_matrix = pd.read_parquet(config["input_path"]["mask_matrix"])
    
    # Run program
    index_model = 1
    results = []
    for index_model, (train, test) in enumerate(split_dataset(df_data, df_candidates, n_splits=config["cv"]["n_splits"], n_repeats=config["cv"]["n_repeats"], keep_balance=config["cv"]["keep_balance"], cv_config=config["cv"])):
        # Create directory
        make_dirs("{}/cv_{}".format(config["output_path"], index_model))
    
        # Load data
        if config["cv"]["mode"] == "recommendation":
            df_data_train = df_data
            df_data_test = df_candidates
        else:
            df_data_train = df_data.iloc[train]
            df_data_test = df_data.iloc[test]

        # Knowledge extracted from LLM models
        expert_parsers = []
        for data in np.load(config['input_path']['expert_knowledge']):
            expert_parsers.append(
                ExpertParser(name=data[0][2], expert_data=data)
            )
        
        # Train model
        clf = HybridClassifier(
            core_set=config["predicting_variables"], frame_of_discernment=frozenset({"High", "Low"}), n_gram_evidence=2, 
            expert_parsers=expert_parsers, alpha=config["dst"]["alpha"], version=config["dst"]["version"],
            df_mask_matrix=df_mask_matrix, scoring=config["scoring"]
        )
        clf.fit(df_data_train["set_name"].values, df_data_train["Label"].values, spark=spark)
        clf.summary_classifiers
        
        ## Serialize summary of classifiers into file
        json.dump(clf.summary_classifiers, open("{}/cv_{}/classifier.json".format(config["output_path"], index_model), 'w' ))

        # All classifiers
        df_evaluation = df_data_test.copy()
        y_preds, final_decisions = clf.predict(
            df_data_test["set_name"].values, spark=spark, show_decision=True,
            baseline_predictions=[[0.5, 0.5]] * len (df_data_test)
        )
        df_evaluation, acc_score, acc_sigificant_score, acc_score_sample = summarize_evaluation(df_evaluation, final_decisions, y_preds)
        df_evaluation.to_csv("{}/cv_{}/hybrid.csv".format(config["output_path"], index_model))
        results.append([
            config["name"], config["cv"]["n_splits"], index_model,
            "hybrid", acc_score, acc_sigificant_score,
            acc_score_sample
        ])

        # Single classifier
        for key, classifier in clf.expert_classifiers.items():
            # Create directory
            make_dirs("{}/cv_{}/{}".format(config["output_path"], index_model, key))
            df_evaluation = df_data_test.copy()
            y_preds, final_decisions = classifier.predict(
                df_data_test["set_name"].values, spark=spark, show_decision=True,
                baseline_predictions=[[0.5, 0.5]] * len (df_data_test)
            )
            df_evaluation, acc_score, acc_sigificant_score, acc_score_sample = summarize_evaluation(df_evaluation, final_decisions, y_preds)
            df_evaluation.to_csv("{}/cv_{}/{}.csv".format(config["output_path"], index_model, key))
            results.append([
                config["name"], config["cv"]["n_splits"], index_model,
                key, acc_score, acc_sigificant_score,
                acc_score_sample
            ])
            # Save similarity matrices 
            classifier.save_similarity_matrices("{}/cv_{}/{}".format(config["output_path"], index_model, key))

        # Save results
        df_results = pd.DataFrame(results, columns=["dataset", "train_size", "cv", "method", "acc_score", "acc_sigificant_score", "acc_score_sample"])
        df_results.to_csv("{}/summary.csv".format(config["output_path"]))


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Load configuration file
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # Generate output directory
    output_path = "{}/{}/{}_{}/{}_{}_{}".format(
        config["output_path"], config["name"], config["cv"]["mode"], config["cv"]["n_splits"],
        config["dst"]["version"], config["dst"]["alpha"], config["dst"]["baseline_estimator"]
    )
    print(output_path)
    config["output_path"] = output_path
    make_dirs(output_path)

    # Save the updated configuration file
    with open("{}/config.yml".format(config["output_path"]), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # Run program
    main(config=config, spark=args["spark"])