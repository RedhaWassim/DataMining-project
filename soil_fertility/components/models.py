from typing import Iterable, List
import pandas as pd
from collections import defaultdict
from itertools import combinations


class AssociationRuleMiner:
    def __init__(self, min_support: int = 4, threshold: float = 1):
        self.min_support = min_support
        if threshold > 1 or threshold < 0:
            raise ValueError("The threshold must be between 0 and 1")
        self.threshold = threshold

    def execute(
        self, data: str, transactions_groups: List[str], items_groups: List[str]
    ):
        self.df = data.copy()
        self.transaction_columns = transactions_groups
        self.items_groups = items_groups

        transactions = self.rearrange_data()

        frequent_itemsets, k = self.apriori(transactions["Items"])

        generated_rules = self.generate_association_rules(frequent_itemsets[k])
        self.find_confidence(generated_rules, transactions["Items"])

    def rearrange_data(self) -> pd.DataFrame:
        """this function is responsible for rearranging the data to be in the form of a transactional dataframe

        Returns:
            pd.DataFrame: a transactional dataframe
        """
        # Melt the DataFrame to convert it from wide to long format
        melted_df = pd.melt(
            self.df, id_vars=self.transaction_columns, value_vars=self.items_groups
        )

        # Drop rows with NaN values (assuming NaNs are not valid items in your context)
        melted_df = melted_df.dropna()

        # Group by the transaction column and aggregate items into a set
        result_df = (
            melted_df.groupby(self.transaction_columns[0])
            .agg({"value": lambda x: set(x)})
            .sort_values(by="value", key=lambda x: x.apply(len), ascending=False)
            .rename(
                columns={
                    self.transaction_columns[0]: "Transaction",
                    "value": "Items",
                }
            )
        )

        return result_df

    def generate_items(self, transactions: pd.Series) -> set:
        """this function is responsible for generating the unique items from the transactions

        Args:
            transactions (pd.Series): a list of transactions

        Returns:
            set: a set containing the unique items
        """
        # generate the unique items from the transactions
        items = set()
        for transaction in transactions:
            items.update(transaction)
        return items

    def generate_candidates(self, items, k) -> list[set]:
        """this function is responsible for generating the candidates from the items

        Args:
            items (_type_): a list of items
            k (_type_): a number representing the cardinality of the candidates

        Returns:
            list[set]: a list of candidates
        """
        # generate the candidates from the items
        # the candidates are the combinations of the items
        result = []
        for combination in combinations(items, k):
            result.append(set(combination))
        return result

    def support(self, transactions: Iterable[set], candidate: set) -> int:
        """this function is responsible for calculating the support of a candidate

        Args:
            transactions (Iterable[set]): a list of transactions
            candidate (set): a candidate

        Returns:
            int : the support of the candidate
        """
        # calculate the support of a candidate
        count = 0
        for transaction in transactions:
            if candidate.issubset(transaction):
                count += 1
        return count

    def select_adequat_candidats(
        self, transactions: Iterable[set], candidates: list[set]
    ) -> dict[dict, int]:
        """this function is responsible for selecting the candidates that have a support greater than the minimum support

        Args:
            transactions (Iterable[set]): a list of transactions
            candidates (list[set]): a list of candidates

        Returns:
            dict[dict, int]: a dictionary containing the candidates and their support
        """
        # select the candidates that have a support greater than the minimum support
        candidat_support = defaultdict(int)
        for transaction in transactions:
            for candidat in candidates:
                # if the candidate is a subset of the transaction
                if set(candidat).issubset(transaction):
                    # increment the support of the candidate
                    candidat_support[tuple(candidat)] += 1
        return {
            candidat: support
            for candidat, support in candidat_support.items()
            if support >= self.min_support
        }

    def apriori(self, transactions: Iterable[set]) -> tuple[dict[int, set], int]:
        """this function is responsible for generating the frequent itemsets from the transactions

        Args:
            transactions (Iterable[set]): a list of transactions

        Returns:
            tuple[dict[int, set], int]: a tuple containing the frequent itemsets and the maximum k
        """
        items = self.generate_items(transactions)
        candidates = self.generate_candidates(items, 1)
        candidats_support = self.select_adequat_candidats(transactions, candidates)
        if not candidats_support:
            # if there is no candidate with a support greater than the minimum support return an empty set
            return {1: set()}, 1

        # generate the items from the support counts
        items = self.generate_items(candidats_support)

        k = 2
        frequent_itemsets = {}

        while True:
            # generate the candidates
            candidates = self.generate_candidates(items, k)
            # if there is no candidate break the loop
            if not candidates:
                break
            print("C", k, "=", candidates)

            # select the candidates that have a support greater than the minimum support
            candidats_support = self.select_adequat_candidats(transactions, candidates)
            # if there is no candidate break the loop
            if not candidats_support:
                break
            print("L", k, "=", candidats_support)
            # generate the items from the support counts
            items = self.generate_items(candidats_support)
            frequent_itemsets[k] = items
            k += 1

        return frequent_itemsets, k - 1

    def generate_association_rules(
        self, frequent_itemsets: set
    ) -> defaultdict[frozenset, set]:
        """this function is responsible for generating the association rules from the frequent itemsets

        Args:
            frequent_itemsets (set): a set containing the frequent itemsets

        Returns:
            defaultdict[frozenset, set]: a dictionary containing the association rules
        """
        max_k = len(frequent_itemsets)
        rules = defaultdict(set)

        for k in range(1, max_k):
            transactions = list(combinations(frequent_itemsets, k))
            for j in range(1, max_k):
                candidates_transaction = list(combinations(frequent_itemsets, j))
                for translation in transactions:
                    for candidate in candidates_transaction:
                        if not set(candidate).issubset(translation):
                            rules[frozenset(translation)].add(candidate)
        return rules

    def find_confidence(
        self, rules: dict[frozenset, set], transactions: Iterable[set]
    ) -> None:
        """this function is responsible for finding the confidence of the association rules

        Args:
            rules (dict[frozenset, set]): a dictionary containing the association rules
            transactions (Iterable[set]): a list of transactions
            threshold (float, optional): the threshold if the confidence  . Defaults to 0.5.
        """
        # for each transaction in the rules
        for transaction, candidates in rules.items():
            # for each candidate in the transaction
            for candidate in candidates:
                # calculate the confidence of the candidate
                # the confidence is the support of the union of the candidate and the transaction divided by the support of the transaction
                try:
                    union = transaction.union(candidate)
                    confidence = self.support(transactions, union) / self.support(
                        transactions, transaction
                    )
                    # if the confidence is greater than the threshold print the association rule
                    if confidence >= self.threshold:
                        print(f"{list(transaction)} => {candidate} : {confidence:.2f}")
                except Exception as e:
                    pass
