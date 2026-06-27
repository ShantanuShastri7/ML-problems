import math

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    centroids = initial_centroids[:]
    
    for _ in range(max_iterations):
        pointsPerCentroid = {idx: [] for idx in range(k)}
        
        for point in points:
            closestCentroid = calculateEuclideanDist(point, centroids)
            pointsPerCentroid[closestCentroid].append(point)
            
        new_centroids = []
        for idx in range(k):
            points_list = pointsPerCentroid[idx]

            if not points_list:
                new_centroids.append(centroids[idx])
                continue
            
            num_points = len(points_list)
            dimensions = len(points[0])

            new_centroid = tuple(
                sum(p[dim] for p in points_list) / num_points 
                for dim in range(dimensions)
            )
            new_centroids.append(new_centroid)
        
            if centroids == new_centroids:
            	break
            
        centroids = new_centroids

    final_centroids = [
        tuple(round(val, 4) for val in centroid) 
        for centroid in centroids
    ]
    
    return final_centroids


def calculateEuclideanDist(point: tuple[float, ...], centroids: list[tuple[float, ...]]) -> int:
    dist = []
    for centroid in centroids:
        squared_diff_sum = sum((p - c) ** 2 for p, c in zip(point, centroid))
        d = math.sqrt(squared_diff_sum)
        dist.append(d)
        
    min_val = dist[0]
    index = 0
    
    for i in range(1, len(dist)):
        if dist[i] < min_val:
            index = i
            min_val = dist[i]
            
    return index