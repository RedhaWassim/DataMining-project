from pprint import pprint

s = {
    "metrics": {
        "decision_tree": {
            "micro_average": (
                0.8693181818181818,
                0.8693181818181818,
                0.8693181818181818,
            ),
            "macro_average": (
                0.7003235037949483,
                0.7557422969187675,
                0.7214891436026466,
            ),
            "metrics_per_class": {
                0: {
                    "precision": 0.8421052631578947,
                    "recall": 0.9142857142857143,
                    "f1_score": 0.8767123287671234,
                    "specificity": 0.94,
                },
                1: {
                    "precision": 0.925531914893617,
                    "recall": 0.8529411764705882,
                    "f1_score": 0.8877551020408163,
                    "specificity": 0.8170731707317073,
                },
                2: {
                    "precision": 0.3333333333333333,
                    "recall": 0.5,
                    "f1_score": 0.4,
                    "specificity": 0.9882352941176471,
                },
            },
        }
    }
}

pprint(s)
