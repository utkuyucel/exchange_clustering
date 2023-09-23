import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.express as px



class DataCleaner:

    @staticmethod
    def _handle_thousand_and_decimal_separators(value) -> float:
        if isinstance(value, str):
            return float(value.replace('.', '').replace(',', '.'))
        return value

    @staticmethod
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        decimal_point_columns = [
            'volume_by_visitors',
            'cmc_score',
            'android_rate',
            'ios_rate',
            'sikayetvar_rate',
        ]

        numerical_columns = [
            '3m_avg_visitors',
            'twitter_followers',
            'instagram_followers',
            '24h_volume',
            'eksisozluk_comments',
            'instagram_post_count',
        ] + decimal_point_columns

        for col in numerical_columns:
            data[col] = data[col].apply(lambda x: DataCleaner._handle_thousand_and_decimal_separators(x) if pd.notnull(x) else x)

        return data


class DataProcessor:

    @staticmethod
    def compute_weighted_ratings(data: pd.DataFrame) -> pd.DataFrame:
        # Compute weighted ratings
        data['weighted_android_rate'] = data['android_rate'] * data['android_comments']
        data['weighted_ios_rate'] = data['ios_rate'] * data['ios_comments']
        data['weighted_sikayetvar_rate'] = data['sikayetvar_rate'] * data['sikayetvar_tickets']

        # Drop non-weighted rate columns
        data = data.drop(columns=['android_rate', 'ios_rate', 'sikayetvar_rate'])

        return data


class ClusteringHandler:

    def __init__(self, n_clusters: int = 4, random_state: int = 34):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def cluster_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # KNN imputation
        features = data.drop(columns=['exchange_name']).dropna()
        knn_imputer = KNNImputer(n_neighbors=3)
        imputed_data = knn_imputer.fit_transform(features)

        # Normalizing the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(imputed_data)

        # KMeans clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        clusters = kmeans.fit_predict(scaled_data)

        # Assign the cluster labels
        data.loc[features.index, 'Cluster'] = clusters
        return data

    @staticmethod
    def compute_feature_importances(data, clusters):
        """
        Computes feature importances using Random Forest.

        :param data: DataFrame, feature matrix
        :param clusters: array-like, cluster labels
        :return: sorted feature importances in descending order
        """
        clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(data, clusters)
        feature_importances = clf.feature_importances_

        # Pairing feature names with their importance scores
        features = list(data.columns)
        feature_importance_dict = dict(zip(features, feature_importances))

        # Sorting the features based on importance
        sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

        return sorted_feature_importance

    @staticmethod
    def sort_clusters(data: pd.DataFrame, sort_by: str) -> pd.DataFrame:
        cluster_means = data.groupby('Cluster')[sort_by].mean().sort_values()
        sorted_cluster_mapping = {old_cluster: new_cluster for new_cluster, old_cluster in enumerate(cluster_means.index)}
        data['Cluster'] = data['Cluster'].map(sorted_cluster_mapping)
        return data


class Reporting:

    @staticmethod
    def display_cluster_averages(cluster_averages: pd.DataFrame):
        for cluster_num in cluster_averages.index:
            print(f"------- Cluster {cluster_num} Mean Values -------")
            for column in cluster_averages.columns:
                mean_value = cluster_averages.loc[cluster_num, column]
                # Check if the value is a float for better formatting
                if isinstance(mean_value, float):
                    print(f"{column}: {mean_value:.2f}")
                else:
                    print(f"{column}: {mean_value}")
            print("\n")




def main():
    data = pd.read_csv('exchanges_data.csv')
    data = DataCleaner.clean_data(data)
    data = DataProcessor.compute_weighted_ratings(data)

    clustering_handler = ClusteringHandler()
    data = clustering_handler.cluster_data(data)

    data = ClusteringHandler.sort_clusters(data, '3m_avg_visitors')

    # Displaying cluster averages
    cluster_averages = data.groupby('Cluster').mean()
    Reporting.display_cluster_averages(cluster_averages)

    # Display exchanges in each cluster
    for cluster_num in cluster_averages.index:
        exchanges_in_cluster = data[data['Cluster'] == cluster_num]['exchange_name'].tolist()
        print(f"Cluster {cluster_num}:")
        print(', '.join(exchanges_in_cluster))
        print("\n")

    # Compute Feature Importances
    features = data.drop(columns=['exchange_name', 'Cluster']).dropna()
    clusters = data.loc[features.index, 'Cluster']
    feature_importances = ClusteringHandler.compute_feature_importances(features, clusters)
    print("\nFeature Importances:")
    for feature, importance in feature_importances:
        print(f"{feature}: %{(importance * 100):.2f}")


if __name__ == "__main__":
    main()
