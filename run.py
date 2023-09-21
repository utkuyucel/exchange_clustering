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
        decimal_point_columns = ['3m_avg_visitors', 'twitter_followers', 'instagram_followers', '24h_volume', 'volume_by_visitors', 'cmc_score', 'android_rate', 'ios_rate', 'sikayetvar_rate']
        for col in decimal_point_columns:
            data[col] = data[col].apply(DataCleaner._handle_thousand_and_decimal_separators)
        return data

class DataProcessor:

    @staticmethod
    def compute_weighted_ratings(data: pd.DataFrame) -> pd.DataFrame:
        data['weighted_android_rate'] = data['android_rate'] * data['android_comments']
        data['weighted_ios_rate'] = data['ios_rate'] * data['ios_comments']
        data['weighted_sikayetvar_rate'] = data['sikayetvar_rate'] * data['sikayetvar_tickets']
        return data

class ClusteringHandler:

    def __init__(self, n_clusters: int = 5, random_state: int = 34):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def cluster_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # KNN imputation
        features = data.drop(columns=['exchange_name']).dropna()
        knn_imputer = KNNImputer(n_neighbors=5)
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
        print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    main()
