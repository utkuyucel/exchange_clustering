from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


class DataCleaner:
    """A class used for cleaning the DataFrame."""

    @staticmethod
    def _handle_thousand_and_decimal_separators(value) -> float:
        """Converts the given value to float and handles thousand and decimal separators."""
        if isinstance(value, str):
            return float(value.replace('.', '').replace(',', '.'))
        return value

    @staticmethod
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """Cleans the given DataFrame."""
        decimal_point_columns = [
            'cmc_score', 'android_rate', 'ios_rate', 'sikayetvar_rate'
        ]
        numerical_columns = [
            '3m_avg_visitors', 'twitter_followers', 'instagram_followers',
            'eksisozluk_comments', 'instagram_post_count', 'tweet_count',
            'has_launchpad','has_staking','has_own_token',
        ] + decimal_point_columns

        data[numerical_columns] = data[numerical_columns].applymap(DataCleaner._handle_thousand_and_decimal_separators)
        return data


class DataProcessor:
    """A class used for processing the DataFrame."""

    @staticmethod
    def compute_weighted_ratings(data: pd.DataFrame) -> pd.DataFrame:
        """Computes and returns DataFrame with weighted ratings."""
        data['weighted_android_rate'] = data['android_rate'] * data['android_comments']
        data['weighted_ios_rate'] = data['ios_rate'] * data['ios_comments']
        data['weighted_sikayetvar_rate'] = data['sikayetvar_rate'] * data['sikayetvar_tickets']
        data.drop(columns=['android_rate', 'android_comments','ios_rate', 'ios_comments', 'sikayetvar_rate', 'sikayetvar_tickets'], inplace=True)
        return data


class ClusteringHandler:
    """Handles clustering related operations on DataFrame."""

    def __init__(self, n_clusters: int = 5, random_state: int = 34):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def cluster_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Performs clustering on the given DataFrame."""
        features = data.drop(columns=['exchange_name']).dropna()
        imputed_data = KNNImputer(n_neighbors=3).fit_transform(features)
        scaled_data = StandardScaler().fit_transform(imputed_data)

        clusters = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(scaled_data)
        data.loc[features.index, 'Cluster'] = clusters
        return data

    @staticmethod
    def compute_feature_importances(data, clusters):
        """Computes and returns sorted feature importances using Random Forest."""
        clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(data, clusters)
        feature_importances = dict(zip(list(data.columns), clf.feature_importances_))
        return sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)

    @staticmethod
    def sort_clusters(data: pd.DataFrame, sort_by: str) -> pd.DataFrame:
        """Sorts and returns clusters in DataFrame based on mean values of the specified column."""
        cluster_means = data.groupby('Cluster')[sort_by].mean().sort_values()
        sorted_cluster_mapping = {old_cluster: new_cluster for new_cluster, old_cluster in enumerate(cluster_means.index)}
        data['Cluster'] = data['Cluster'].map(sorted_cluster_mapping)
        return data


class Reporting:
    """Handles the reporting of cluster information."""

    @staticmethod
    def display_cluster_averages(cluster_averages: pd.DataFrame):
        """Displays mean values for each cluster."""
        for cluster_num in cluster_averages.index:
            print(f"------- Cluster {cluster_num} Mean Values -------")
            for column in cluster_averages.columns:
                mean_value = cluster_averages.loc[cluster_num, column]
                print(f"{column}: {mean_value:.2f}" if isinstance(mean_value, float) else f"{column}: {mean_value}")
            print("\n")

    @staticmethod
    def calculate_improvements(exchange_name: str, data: pd.DataFrame, cluster_averages: pd.DataFrame) -> None:
        exchange_data = data.loc[data['exchange_name'] == exchange_name]
        if exchange_data.empty:
            print("Invalid Exchange Name")
            return
        
        exchange_cluster = exchange_data.iloc[0]['Cluster']
        if exchange_cluster + 1 not in cluster_averages.index:
            print(f"{exchange_name} is already in the highest cluster")
            return
        
        current_cluster_mean = cluster_averages.loc[exchange_cluster]
        higher_cluster_mean = cluster_averages.loc[exchange_cluster + 1]

        improvements = higher_cluster_mean - exchange_data.iloc[0]
        print(f"To move {exchange_name} to cluster {exchange_cluster + 1}, the following improvements are needed:")
        for feature, value in improvements.items():
            if feature != 'Cluster' and value > 0:
                print(f"{feature}: {value:.2f}")

    @staticmethod
    def plot_feature_importances(feature_importances: list[dict]):
        """Plot the feature importances using Plotly."""
        feature_importance_df = pd.DataFrame(feature_importances, columns=['Feature', 'Importance'])
        fig = px.bar(feature_importance_df, x='Feature', y='Importance',
                     title='Feature Importances',
                     labels={'Importance': 'Importance (%)'},
                     text='Importance')
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(
            yaxis=dict(tickformat='%'),
            xaxis=dict(tickangle=-45)  # Rotate x-axis labels by 45 degrees
        )
        fig.show()

                
    @staticmethod
    def plot_clusters_with_exchanges(data: pd.DataFrame):
        """Plot clusters with exchanges using a treemap."""
      
        data_copy = data.copy()
        data_copy['dummy'] = 'Exchanges'
        
        cluster_sizes = data_copy.groupby('Cluster')['3m_avg_visitors'].mean().to_dict()
        
        adjusted_cluster_sizes = {cluster: 1 + np.log(size) for cluster, size in cluster_sizes.items()}
        
        data_copy['size'] = data_copy['Cluster'].map(adjusted_cluster_sizes)
        
        unique_clusters = sorted([cluster for cluster in data_copy['Cluster'].dropna().unique() if not pd.isna(cluster)], key=lambda x: -x)
        data_copy['Cluster'] = pd.Categorical(data_copy['Cluster'], categories=unique_clusters, ordered=True)
        
        data_filtered = data_copy.dropna(subset=['Cluster', 'exchange_name'])
        
        fig = px.treemap(data_filtered, path=['Cluster', 'dummy', 'exchange_name'], values='size', title='Exchanges in Clusters')
        fig.show()


def main():
    data = pd.read_csv('exchanges_data.csv')
    data = DataCleaner.clean_data(data)
    data = DataProcessor.compute_weighted_ratings(data)

    clustering_handler = ClusteringHandler()
    data = clustering_handler.cluster_data(data)
    data = ClusteringHandler.sort_clusters(data, '3m_avg_visitors')

    cluster_averages = data.groupby('Cluster').mean()
    Reporting.display_cluster_averages(cluster_averages)
    Reporting.plot_clusters_with_exchanges(data)

    # for cluster_num in cluster_averages.index:
    #     exchanges_in_cluster = data[data['Cluster'] == cluster_num]['exchange_name'].tolist()
    #     print(f"Cluster {cluster_num}:\n", ', '.join(exchanges_in_cluster), "\n")

    features = data.drop(columns=['exchange_name', 'Cluster']).dropna()
    clusters = data.loc[features.index, 'Cluster']
    feature_importances_list = ClusteringHandler.compute_feature_importances(features, clusters)
    
    # Create DataFrame from feature_importances and plot using Plotly
    feature_importances_df = pd.DataFrame(feature_importances_list, columns=['Feature', 'Importance'])
    Reporting.plot_feature_importances(feature_importances_df)

    print("\n################################################################\n")

    exchange_name = "Bitlo"
    Reporting.calculate_improvements(exchange_name, data, cluster_averages)


if __name__ == "__main__":
    main()
