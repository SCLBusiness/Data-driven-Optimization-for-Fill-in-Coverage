"""
 An optimization program corresponds to my proposed framework based on DBSCAN
 Written by: Sichong Liao
 Version 22th August
 Date 22th August 2023
 Postscript: An individual program to generate maps. Including the function of the DataFrameWriter.py
"""
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
import pandas as pd
import math
import random
from math import log10,sin,pow,sqrt
from queue import Queue
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib.patches import RegularPolygon
import matplotlib.patches as patches
from sklearn.metrics import silhouette_score,silhouette_samples
from sklearn.cluster import DBSCAN


def dbscan(X, eps, min_samples):
    # 初始化标记数组，-1表示噪声点，0表示未访问过的样本点
    labels = np.zeros(len(X), dtype=int)
    cluster_id = 0

    for i in range(len(X)):
        if labels[i] != 0:
            continue

        # 获取样本的邻居点
        neighbors = find_neighbors(X, i, eps)

        # 如果邻居点数量小于min_samples，则标记为噪声点
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        cluster_id += 1
        labels[i] = cluster_id

        # 扩展聚类
        expand_cluster(X, labels, i, neighbors, cluster_id, eps, min_samples)

    return labels

def find_neighbors(X, i, eps):
    neighbors = []
    for j in range(len(X)):
        if np.linalg.norm(X[i] - X[j]) < eps:
            neighbors.append(j)
    return neighbors

def expand_cluster(X, labels, i, neighbors, cluster_id, eps, min_samples):
    for neighbor in neighbors:
        if labels[neighbor] == -1:
            labels[neighbor] = cluster_id
        elif labels[neighbor] == 0:
            labels[neighbor] = cluster_id
            neighbor_neighbors = find_neighbors(X, neighbor, eps)
            if len(neighbor_neighbors) >= min_samples:
                neighbors.extend(neighbor_neighbors)

def haversine_distance(lat1, lon1, lat2, lon2):

    # 将经纬度转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 应用Haversine公式计算距离
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    distance = 6371 * c * 1000  # 地球半径为6371公里，乘以1000转换为米

    return distance
#找到一个正六边形的相邻其他正六边形的中点
def find_adjacent_hexagons(center_x, center_y, R):
    neighbors = []

    # Upper-Left Hexagon
    neighbors.append((round(center_x - (3*R/2),3), round(center_y - R*pow(3,0.5) / 2,3)))

    # Upper-Right Hexagon
    neighbors.append((round(center_x +(3*R/2),3), round(center_y - R*pow(3,0.5) / 2,3)))

    # Upper Hexagon
    neighbors.append((round(center_x,3) , round(center_y- pow(3,0.5)* R ,3)))

    # Lower-Right Hexagon
    neighbors.append((round(center_x +(3*R/2),3),  round(center_y+ R*pow(3,0.5) / 2,3)))

    # Lower-Left Hexagon
    neighbors.append((round(center_x - (3*R/2),3),round(center_y +R*pow(3,0.5) / 2,3)))

    # Lower Hexagon
    neighbors.append((round(center_x,3) , round(center_y+pow(3,0.5)* R ,3)))

    return neighbors

#判断是否在范围内
def is_within_range(point,xmin,xmax,ymin,ymax):
    if xmin <= point[0] <= xmax and ymin <= point[1] <= ymax:
        return True
    return False
def is_not_in_center_coordinates(point,center_coordinates):
    for p in center_coordinates:

        if p[0]==point[0] and p[1]==point[1]:
            return False
        if p[0]==point[0] and abs(p[1]-point[1])<=0.002:
            return False
        if p[1]==point[1] and abs(p[0]-point[0])<=0.002:
            return False
    return True

def cost231_hata_model( distance ):
    frequency = 2100
    hb = 30
    hr = 1.5
    alpha = (1.1 * math.log10(frequency) - 0.7) * hr - (1.56 * math.log10(frequency) - 0.8)
    path_loss = 46.3 + 33.9 * math.log10(frequency) - 13.82 * math.log10(hb) - alpha + (
                44.9 - 6.55 * math.log10(hb)) * math.log10(distance)
    return path_loss

def RMa_LOS_d2d(d2d):
    fc=0.5
    Hbs=21
    Hue=1.5
    Hbu=8
    d3d=sqrt(d2d**2+(Hbs-Hue)**2)
    dbp=2*3.1415926*Hbs*Hue*fc*10**9/(3*10**8)
    PL1=20*log10(40*3.1415926*d3d*fc/3)+min(0.03*pow(Hbu,1.72),10)*log10(d3d)\
    -min(0.044*pow(Hbu,1.72),14.77)+0.002*log10(Hbu)*d3d
    PL2=PL1+40*log10(d3d/dbp)

    if d2d<=dbp:
        PL=PL1
    if d2d>dbp:
        PL=PL2
    CL = log10(4 + 2 * fc)
    return PL+CL

def calculate_SBS_received_power(transmit_power, distance):
    transmit_power_dbm = 10 * math.log10(transmit_power)
    path_loss = RMa_LOS_d2d(distance)
    received_power_dbm = transmit_power_dbm - path_loss
    return received_power_dbm

def calculate_MBS_received_power(transmit_power, distance):
    transmit_power_dbm = 10 * math.log10(transmit_power)
    path_loss = cost231_hata_model(distance)
    received_power_dbm = transmit_power_dbm - path_loss
    return received_power_dbm

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def is_high_traffic_center(center_x, center_y, points, radius):
    count = 0
    for point in points:
        distance = euclidean_distance(center_x, center_y, point[0], point[1])
        if distance <= radius:
            count += 1
    return count


data = pd.read_csv('./data/Los_Angeles_136k.csv', header=None, names=['time', 'latitude', 'longitude'])[0:6000]
USER = []
for i in range(len(data)):
    USER.append([data['latitude'][i],data['longitude'][i]])
USER=np.array(USER)
lat_min, lat_max = data['latitude'].min()-0.01, data['latitude'].max()+0.01
lon_min, lon_max = data['longitude'].min()-0.01, data['longitude'].max()+0.01
# rectangle dimensions
latDelta = lat_max - lat_min
lonDelta = lon_max - lon_min

MBS_R_km=2
MBS_R=MBS_R_km/111

center_point=(round((lat_max-lat_min)/2+lat_min,3),round((lon_max-lon_min)/2+lon_min,3))

MBS_coordinates=[center_point]
adjacent_hexagons = find_adjacent_hexagons(center_point[0], center_point[1], MBS_R)

queue_hexagons = Queue()

for point in adjacent_hexagons:
    queue_hexagons.put(point)
while  not queue_hexagons.empty():
    point=queue_hexagons.get()
    if  is_within_range(point,lat_min, lat_max,lon_min, lon_max) and is_not_in_center_coordinates(point,MBS_coordinates):
            # print(point)
            MBS_coordinates.append(point)
            adjacent_hexagons=find_adjacent_hexagons(point[0],point[1],MBS_R)
            for point in adjacent_hexagons:
                if is_not_in_center_coordinates(point,MBS_coordinates) and is_within_range(point, lat_min, lat_max,lon_min, lon_max):
                        queue_hexagons.put(point)
# print(MBS_coordinates)

SIG=[]
MBS_USER=[]
for j in range(len(MBS_coordinates)):
    MBS_USER.append([])
transmit_power=30
for i in range(len(USER)):
    signal_intensity=-500
    num=0
    for j in range(len(MBS_coordinates)):
        distance=abs(haversine_distance(USER[i][0], USER[i][1], MBS_coordinates[j][0], MBS_coordinates[j][1]))/1000
        if signal_intensity<calculate_MBS_received_power(transmit_power, distance):
            signal_intensity=calculate_MBS_received_power(transmit_power, distance)
            num=j
    MBS_USER[num].append(USER[i])
    SIG.append(signal_intensity)

# DBSCAN
SBS_num = 13
SBS_R_km = 0.5
SBS_R = SBS_R_km/111
eps = (MBS_R_km/2)/111
min_samples = 40
DBSCAN = DBSCAN(eps=eps, min_samples=min_samples).fit(USER)
# labels = dbscan(USER, eps, min_samples)
labels = DBSCAN.labels_

cluster_centers = []
for label in set(labels):
    if label != -1:
        cluster = USER[labels == label]
        center = np.mean(cluster, axis=0)
        cluster_centers.append(center)

print(len(cluster_centers))
k = len(cluster_centers)

dist_centroids={}
for center_x,center_y in cluster_centers:
    dist_centroids[(center_x, center_y)]=[]

radius = SBS_R
min_count = 20
SBS_centroids=[]
del_centroids=[]
hot_points=[]

for center_x, center_y in cluster_centers:
    point_density=is_high_traffic_center(center_x, center_y, USER, radius)
    dist_centroids[(center_x, center_y)].append(point_density)
    if point_density>min_count:
        hot_points.append((center_x, center_y))
    else:
        del_centroids.append([center_x, center_y])

silhouette_scores = silhouette_score(USER, labels)
print(silhouette_scores)
silhouette_sample=silhouette_samples (USER, labels)

cluster_scores = []
for cluster_label in range(k):
    cluster = []
    for i in range(len(USER)):
        if labels[i]==cluster_label:
            cluster.append(silhouette_sample[i])
    cluster=np.array(cluster)
    mean_cluster_silhouette_score = np.mean(cluster)
    cluster_scores.append(mean_cluster_silhouette_score)
    dist_centroids[(cluster_centers[cluster_label][0], cluster_centers[cluster_label][1])].append(mean_cluster_silhouette_score)
print(cluster_scores)

centroids_to_centroids=np.zeros(k)
for i in range(k):
    if cluster_centers[i][0]==0 and cluster_centers[i][1]==0:
        continue
    dis=0
    for j in range(k):
        if cluster_centers[i][0] == 0 and cluster_centers[i][1] == 0:
            continue
        dis=dis+euclidean_distance(cluster_centers[i][0], cluster_centers[i][1], cluster_centers[j][0], cluster_centers[j][1])
    centroids_to_centroids[i]=dis
    dist_centroids[(cluster_centers[i][0], cluster_centers[i][1])].append(dis)
print(centroids_to_centroids)

SBS_ranking = [(key, dist_centroids[key][2]) for key in dist_centroids] # Create a list of tuples containing the key and silhouette score

# First, create a set of (center_x, center_y) pairs from del_centroids for efficient lookup
del_centroids_set = {(center_x, center_y) for center_x, center_y in del_centroids}

# Remove points with the same (center_x, center_y) values from SBS_ranking
SBS_ranking = [entry for entry in SBS_ranking if entry[0] not in del_centroids_set]

SBS_ranking_all_parameter_output = [] # Create a list to hold the sorted dist_centroids values

for key, _ in SBS_ranking: # Iterate through the sorted keys and access the associated information in dist_centroids
    center_x, center_y = key  # Unpack the key
    point_density, mean_silhouette_score, _ = dist_centroids[key]  # Unpack the values. And add an additional underscore _ to the unpacking of the value in the loop. This underscore acts as a placeholder for the value you don't need to unpack. This should help resolve the "too many values to unpack" error
    SBS_ranking_all_parameter_output.append((center_x, center_y, point_density, mean_silhouette_score))

SBS_ranking_all_parameter_output.sort(key=lambda x: x[3], reverse=True)  # Sort based on mean_cluster_silhouette_score and related information will follow the movement

if SBS_num < len(SBS_ranking_all_parameter_output):
    for i in range(SBS_num, len(SBS_ranking_all_parameter_output)):
        center_x = SBS_ranking_all_parameter_output[i][0]
        center_y = SBS_ranking_all_parameter_output[i][1]
        del_centroids.append((center_x, center_y))
    SBS_ranking_all_parameter_output=SBS_ranking_all_parameter_output[0:SBS_num] # slicing the first `SBS_num` elements from the `SBS_ranking_all_parameter_output` list. This will ensure that `SBS_ranking_all_parameter_output` contains the required number of elements with all the information.
else:
    print('Amount of Fill-in Coverage Candidates is lesser than SBS_num', len(SBS_ranking_all_parameter_output))

# DataFrame
SBS_centroids_DF=pd.DataFrame(SBS_ranking_all_parameter_output,
                              columns=['Latitude', 'Longitude', 'Point Density', 'Mean Silhouette']) # 排列顺序要遵从元素被append进数据结构的顺序
SBS_centroids_DF.to_csv('Fill-in_Coverage_Candidates_based_on_DBSCAN.csv', index=False)

transmit_power=10
DBSCAN_SIG=[]
for i in range(len(USER)):
    signal_intensity=SIG[i]

    for j in range(len(SBS_centroids)):
        distance=abs(haversine_distance(USER[i][0], USER[i][1], SBS_centroids[j][0], SBS_centroids[j][1]))
        if signal_intensity<calculate_SBS_received_power(transmit_power, distance):
            signal_intensity=calculate_SBS_received_power(transmit_power, distance)

    DBSCAN_SIG.append(signal_intensity)

signal_power_interval = {'135':0,'125':0,'115': 0, '105': 0, '95': 0, '85': 0, '75': 0, '65': 0, '55': 0, '45': 0, '35': 0}
res=[]
for sig in DBSCAN_SIG:
    for key in signal_power_interval.keys():
        if sig<=0-int(key):
            signal_power_interval[key]=signal_power_interval[key]+1
            break

for key,value in signal_power_interval.items():
    res.append(value/len(SIG))
    #print(0-int(key),' ',value/len(SIG))

print(res)
DBSCAN_SIG=np.array(DBSCAN_SIG)
print(np.mean(DBSCAN_SIG))

cluster_centers=np.array(cluster_centers)
SBS_random_centroids=cluster_centers[random.sample(range(1, k), SBS_num)]

transmit_power=10
DBSCAN_SIG=[]
for i in range(len(USER)):
    signal_intensity=SIG[i]
    for j in range(len(SBS_random_centroids)):
        distance=abs(haversine_distance(USER[i][0], USER[i][1], SBS_random_centroids[j][0], SBS_random_centroids[j][1]))
        if signal_intensity<calculate_SBS_received_power(transmit_power, distance):
            signal_intensity=calculate_SBS_received_power(transmit_power, distance)
    DBSCAN_SIG.append(signal_intensity)


signal_power_interval = {'135':0,'125':0,'115': 0, '105': 0, '95': 0, '85': 0, '75': 0, '65': 0, '55': 0, '45': 0, '35': 0}
res=[]
for sig in DBSCAN_SIG:
    for key in signal_power_interval.keys():
        if sig<=0-int(key):
            signal_power_interval[key]=signal_power_interval[key]+1
            break

for key,value in signal_power_interval.items():
    res.append(value/len(SIG))
    #print(0-int(key),' ',value/len(SIG))

print(res)
DBSCAN_SIG=np.array(DBSCAN_SIG)
print(np.mean(DBSCAN_SIG))
# Plot clustering results
# Draw picture Macro base station and user distribution map, red represents users, blue represents base stations
fig, ax = plt.subplots(figsize=(3.5, 3.5 * lonDelta / latDelta))

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] # 'b': Blue，'g': Green，'r': Red，'c': Cyan (a mix of blue and green)，'m': Magenta (a mix of blue and red)，'y': Yellow (a mix of green and red)，'k': Black
for i, point in enumerate(USER):
    cluster = labels[i]
    # Check if the point is noise (cluster = -1)
    if cluster == -1:
        color = 'k'
        plt.scatter(point[0], point[1], c=color, marker='o', s=5, alpha=0.3)
    else:
        color = colors[cluster % (len(colors) - 1)]  # Subtract 1 to exclude 'k' from the cycle for non-noise points
        plt.scatter(point[0], point[1], c=color, marker='o',s=5)

# Create a rectangle patch
rect = patches.Rectangle((lat_min, lon_min), latDelta, lonDelta, linewidth=1, edgecolor='r', facecolor='none')
# Add the rectangle to the plot
ax.add_patch(rect)

# MBS
for center_x, center_y in MBS_coordinates:
    # Create the regular hexagon centered at (center_x, center_y)

    hexagon = RegularPolygon((center_x, center_y), numVertices=6, radius=MBS_R, edgecolor='black',
                             facecolor=(0.6, 0.6, 0.6, 0.01), orientation=np.radians(30), alpha=0.1)
    plt.scatter(center_x, center_y, color='black', marker='^', s=10, alpha=0.6)
    # Add the hexagon to the plot
    ax.add_patch(hexagon)

# Draw Fill-in Coverage candidates
for center_x, center_y, _, _ in SBS_ranking_all_parameter_output:  # add two underscore placeholders _ after center_y in the unpacking part to match the expected number of values. This should help resolve the "too many values to unpack" error
    # Create the regular hexagon centered at (center_x, center_y)
    if center_x>lat_max or center_x<lat_min or center_y>lon_max or center_y<lon_min:
        continue
    circle = plt.Circle((center_x, center_y), SBS_R, color='red', fill=False)
    plt.gca().add_patch(circle)
    plt.scatter(center_x, center_y, color='c', marker='^', s=10)

# Draw unavailable hot spot candidates
for center_x, center_y in del_centroids:
    # Create the regular hexagon centered at (center_x, center_y)
    if center_x>lat_max or center_x<lat_min or center_y>lon_max or center_y<lon_min:
        continue
    circle = plt.Circle((center_x, center_y), SBS_R, color='black', fill=False)
    plt.gca().add_patch(circle)
    plt.scatter(center_x, center_y, color='c', marker='^', s=10)

# Set the axis limits based on the maximum and minimum coordinates
x_coordinates, y_coordinates = zip(*MBS_coordinates)
plt.xlim(min(x_coordinates) - MBS_R * 1.5, max(x_coordinates) + MBS_R * 1.5)
plt.ylim(min(y_coordinates) - MBS_R * 1.5, max(y_coordinates) + MBS_R * 1.5)

# Set axis labels
plt.xlabel('Latitude')
plt.ylabel('Longitude')

# Show the plot
plt.axis('equal')
plt.grid(True)
#plt.title('DBSCAN Clustering')
plt.savefig('Identified fill-in coverage candidates based on DBSCAN.png', dpi=300, bbox_inches='tight')
plt.show()

