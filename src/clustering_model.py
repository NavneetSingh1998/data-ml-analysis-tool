import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

class ClusteringModel:
    def __init__(self, model_type='kmeans', n_clusters=3, random_state=42):
        self.model_type = model_type
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = self._initialize_model()
        self.labels = None
        self.scaler = StandardScaler()

    def _initialize_model(self):
        if self.model_type == 'kmeans':
            return KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        elif self.model_type == 'hierarchical':
            return AgglomerativeClustering(n_clusters=self.n_clusters)
        elif self.model_type == 'dbscan':
            return DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError('Model type not recognized')

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.labels = self.model.fit_predict(X_scaled)
        return self.labels

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        if self.model_type == 'kmeans':
            return self.model.predict(X_scaled)
        else:
            return self.model.labels_

    def evaluate(self, X):
        X_scaled = self.scaler.fit_transform(X)
        labels = self.fit(X)
        
        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        
        return {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'n_clusters': len(set(labels))
        }

    def find_optimal_clusters(self, X, max_clusters=10):
        X_scaled = self.scaler.fit_transform(X)
        scores = []
        
        for n in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=self.random_state)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            scores.append(score)
        
        optimal_clusters = np.argmax(scores) + 2
        return optimal_clusters, scores
