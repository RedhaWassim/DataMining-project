from itertools import combinations
from collections import Counter
import pandas as pd
import math
class Metrics:
    def __init__(self, confidence=0, cosine=0, lift=0, all_confidence=0, jaccard=0, kulczynski=0, max_confidence=0):
        self.confidence = confidence
        self.cosine = cosine
        self.lift = lift
        self.all_confidence = all_confidence
        self.jaccard = jaccard
        self.kulczynski = kulczynski
        self.max_confidence = max_confidence

    @classmethod
    def from_rules(cls, itemset, antecedent, support, parent):
        metrics = cls()
        metrics.calculate_all(itemset, antecedent, support, parent)
        return metrics

    def calculate_all(self, itemset, antecedent, support, parent):
        self.confidence = support / parent._get_support(antecedent)
        self.cosine = self._calculate_cosine(itemset, antecedent, parent)
        self.lift = self._calculate_lift(itemset, antecedent, parent)
        self.jaccard = self._calculate_jaccard(itemset, antecedent, parent)
        self.kulczynski = self._calculate_kulczynski(itemset, antecedent, parent)
        self.max_confidence = self._calculate_max_confidence(itemset, antecedent, parent)
        self.all_confidence = self._calculate_all_confidence(itemset, antecedent, parent)

    def _calculate_cosine(self, itemset, antecedent, parent) -> float:
        return parent._get_support(itemset | antecedent) / math.sqrt(parent._get_support(itemset) * parent._get_support(antecedent))

    def _calculate_lift(self, itemset, antecedent, parent) -> float:
        confidence = parent._get_support(itemset | antecedent) / parent._get_support(antecedent)
        return confidence / parent._get_support(itemset)

    def _calculate_jaccard(self, itemset, antecedent, parent) -> float:
        join_support = parent._get_support(itemset | antecedent)
        return join_support / (parent._get_support(itemset) + parent._get_support(antecedent) - join_support)

    def _calculate_kulczynski(self, itemset, antecedent, parent) -> float:
        join_support = parent._get_support(itemset | antecedent)
        return ((join_support / parent._get_support(itemset)) + (join_support / parent._get_support(antecedent))) / 2

    def _calculate_max_confidence(self, itemset, antecedent, parent) -> float:
        join_support = parent._get_support(itemset | antecedent)
        return max(parent._get_support(itemset) / join_support, parent._get_support(antecedent) / join_support)

    def _calculate_all_confidence(self, itemset, antecedent, parent) -> float:
        return parent._get_support(itemset | antecedent) / max(parent._get_support(itemset), parent._get_support(antecedent))


class Rule:
    def __init__(self, antecedent, consequent, metrics):
        self.antecedent = antecedent
        self.consequent = consequent
        self.metrics = metrics




class MyApriori:
    def __init__(self, min_support, min_confidence):
        self.df = None
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = []
        self.transactions = None
        self.transaction_columns = None
        self.items_groups = None
        self.show = False
        self.rules = []

    def reset(self):
        self.df = None
        self.frequent_itemsets = []
        self.transactions = None
        self.transaction_columns = None
        self.items_groups = None
        self.show = False
        self.rules = []


    def set_parameters(self, min_support, min_confidence):
        
        self.reset()
        self.min_support = min_support
        self.min_confidence = min_confidence



    def fit(self, input_df: pd.DataFrame, transaction_columns: list[str], items_groups: list[str], show : bool = False) -> None:
        self.frequent_itemsets = []
        self.df = input_df.copy()
        self.transaction_columns = transaction_columns
        self.items_groups = items_groups
        self.show = show
        self.transactions = self.rearrange_data()
        self._generate_frequent_items()
        self._generate_rules()



    def rearrange_data(self) -> pd.DataFrame:
        """This function is responsible for rearranging the data to be in the form of a transactional dataframe.

        Returns:
            pd.DataFrame: A transactional dataframe.
        """
        # Melt the DataFrame to convert it from wide to long format
        melted_df = pd.melt(
            self.df, id_vars=self.transaction_columns, value_vars=self.items_groups
        )

        # Drop rows with NaN values (assuming NaNs are not valid items in your context)
        melted_df = melted_df.dropna()

        # Group by all transaction columns and aggregate items into a set
        result_df = (
            melted_df.groupby(self.transaction_columns)
            .agg({"value": lambda x: set(x)})
            .sort_values(by="value", key=lambda x: x.apply(len), ascending=False)
            .rename(
                columns={
                    tuple(self.transaction_columns): "Transaction",
                    "value": "Items",
                }
            )
        )

        return result_df
    
    def _generate_frequent_items(self) :
        """this function is responsible for generating the items from the transactions

        Args:
            transactions (pd.Series): a list of transactions

        Returns:
            set: a set containing the unique items
        """
        self._generate_one_itemset()
        
        if self.show:
            print("L1", self.frequent_itemsets[0])
            print('\n')

        k=2
        while True:
            frequent_items = self._generate_candidates(k)
            if not frequent_items:
                break

            if self.show:
                print(f"L{k}", frequent_items)
                print('\n')

            self.frequent_itemsets.append(frequent_items)
            k += 1

    def _generate_one_itemset(self):
        """Generate the first item set containing unique items."""

        self.item_counts = Counter()

        for itemset in self.transactions['Items']:
            self.item_counts.update(itemset)

        n = len(self.transactions)

        frequent_itemsets = [
            (frozenset([item]), support / n)
            for item, support in self.item_counts.items()
            if support / n >= self.min_support
        ]

        self.frequent_itemsets.append(frequent_itemsets)


    
    def _generate_candidates(self, k):
        itemsets = self.frequent_itemsets[k - 2]
        frequent_itemsets = []

        for i, (itemset1, _) in enumerate(itemsets):
            for _, (itemset2, _) in enumerate(itemsets[i + 1:]):
                union = itemset1 | itemset2

                if len(union) == k and union not in (itemset for itemset, _ in frequent_itemsets):
                    support = self._get_support(union)

                    if support >= self.min_support:
                        frequent_itemsets.append((union, support))

        return frequent_itemsets

    
    def _get_support(self, itemset : set) -> float:
        """This function is responsible for calculating the support of an itemset

        Args:
            itemset (set): the itemset

        Returns:
            float: the support of the itemset
        """
        count = 0
        total_transactions = len(self.transactions)
        
        for itemset2 in self.transactions['Items']:
            if itemset.issubset(itemset2):
                count += 1
        
        support = count / total_transactions
        
        return support
    
    def _generate_rules(self):
        """This function is responsible for generating the rules from the frequent itemsets.
        """
        self.rules = []
        
        for itemset in self.frequent_itemsets:
            for item, support in itemset:
                if len(itemset) < 2:
                    continue
                self._generate_rules_from_itemset(item, support)

    def _generate_rules_from_itemset(self,itemset : set, support : float):
        """This function is responsible for generating the rules from an itemset.

        Args:
            item (set): the itemset
            support (float): the support of the itemset
        """

        for antecedent in self._get_antecedents(itemset):
            antecedent_support = self._get_support(antecedent)
            if support / antecedent_support >= self.min_confidence:
                if self.show:
                    print(f"{antecedent} => {itemset - antecedent} (Conf: {support / antecedent_support:.2f}, Supp: {support:.2f})")
                self.rules.append((antecedent, itemset - antecedent, support / antecedent_support))

        
    def _get_antecedents(self,itemset: set) -> list[tuple]:
        antecedents = []

        for i in range(1, len(itemset)):
            current_antecedents = combinations(itemset, i)
            current_antecedents = {frozenset(antecedent) for antecedent in current_antecedents}
            antecedents.extend(current_antecedents)

        return antecedents


    def _generate_rules_from_itemset(self, itemset: set, support: float):
        """Generate rules from an itemset.

        Args:
            itemset (set): the itemset
            support (float): the support of the itemset
        """

        for antecedent in self._get_antecedents(itemset):
            antecedent_support = self._get_support(antecedent)

            metrics = Metrics.from_rules(itemset, antecedent, support, self)

            if metrics.confidence >= self.min_confidence:
                rule = Rule(antecedent, itemset - antecedent, metrics)

                if self.show:
                    print(f"{antecedent} => {itemset - antecedent} {metrics.confidence} (Supp: {support:.2f})")

                self.rules.append(rule)

    def get_rules(self, metric='confidence') -> pd.DataFrame:
        """This function is responsible for generating the strong rules from the rules.

        Args:
            metric (str): The metric to use for sorting rules (default: 'confidence').

        Returns:
            pd.DataFrame: A DataFrame containing the strong rules.
        """
        sorting_functions = {
            'confidence': lambda: sorted(self.rules, key=lambda x: x.metrics.confidence, reverse=True),
            'cosine': lambda: sorted(self.rules, key=lambda x: x.metrics.cosine, reverse=True),
            'lift': lambda: sorted(self.rules, key=lambda x: x.metrics.lift, reverse=True),
            'all_confidence': lambda: sorted(self.rules, key=lambda x: x.metrics.all_confidence, reverse=True),
            'jaccard': lambda: sorted(self.rules, key=lambda x: x.metrics.jaccard, reverse=True),
            'kulczynski': lambda: sorted(self.rules, key=lambda x: x.metrics.kulczynski, reverse=True),
            'max_confidence': lambda: sorted(self.rules, key=lambda x: x.metrics.max_confidence, reverse=True),
        }

        if metric in sorting_functions:
            sorted_rules = sorting_functions[metric]()
            rule_data = [(rule.antecedent, rule.consequent, rule.metrics.confidence, rule.metrics.cosine,
                          rule.metrics.lift, rule.metrics.all_confidence, rule.metrics.jaccard,
                          rule.metrics.kulczynski, rule.metrics.max_confidence) for rule in sorted_rules]
            columns = ['Antecedent', 'Consequent', 'Confidence', 'Cosine', 'Lift', 'All Confidence', 'Jaccard', 'Kulczynski', 'Max Confidence']
            strong_rules_df = pd.DataFrame(rule_data, columns=columns)
            return strong_rules_df
        else:
            raise ValueError(f'metric should be one of {", ".join(sorting_functions.keys())}')
        
    def predict(self, items: list[str], metric='confidence'):
        """Predicts the consequents based on the provided items using the specified metric.

        Args:
            items (list[str]): The list of items.
            metric (str): The metric to use for prediction (default: 'confidence').

        Returns:
            List[Tuple[set, float]]: A list of predictions containing consequent sets and their corresponding metric values.
        """
        prediction_functions = {
            'confidence': self._predict_confidence,
            'cosine': self._predict_cosine,
            'lift': self._predict_lift,
            'all_confidence': self._predict_all_confidence,
            'jaccard': self._predict_jaccard,
            'kulczynski': self._predict_kulczynski,
            'max_confidence': self._predict_max_confidence,
        }

        if metric in prediction_functions:
            return prediction_functions[metric](items)
        else:
            raise ValueError(f'metric should be one of {", ".join(prediction_functions.keys())}')

    def _predict_confidence(self, items: list[str]):
        items = set(items)
        predictions = []
        for rule in self.rules:
            if rule.antecedent == items:
                predictions.append((rule.consequent, rule.metrics.confidence))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def _predict_cosine(self, items: list[str]):
        items = set(items)
        predictions = []
        for rule in self.rules:
            if rule.antecedent == items:
                predictions.append((rule.consequent, rule.metrics.cosine))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def _predict_lift(self, items: list[str]):
        items = set(items)
        predictions = []
        for rule in self.rules:
            if rule.antecedent == items:
                predictions.append((rule.consequent, rule.metrics.lift))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def _predict_all_confidence(self, items: list[str]):
        items = set(items)
        predictions = []
        for rule in self.rules:
            if rule.antecedent == items:
                predictions.append((rule.consequent, rule.metrics.all_confidence))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def _predict_jaccard(self, items: list[str]):
        items = set(items)
        predictions = []
        for rule in self.rules:
            if rule.antecedent == items:
                predictions.append((rule.consequent, rule.metrics.jaccard))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def _predict_kulczynski(self, items: list[str]):
        items = set(items)
        predictions = []
        for rule in self.rules:
            if rule.antecedent == items:
                predictions.append((rule.consequent, rule.metrics.kulczynski))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def _predict_max_confidence(self, items: list[str]):
        items = set(items)
        predictions = []
        for rule in self.rules:
            if rule.antecedent == items:
                predictions.append((rule.consequent, rule.metrics.max_confidence))
        return sorted(predictions, key=lambda x: x[1], reverse=True)


