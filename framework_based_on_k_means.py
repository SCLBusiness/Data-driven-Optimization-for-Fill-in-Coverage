"""
 An optimization program corresponds to my proposed framework based on K-means
 Written by: Sichong Liao
 Version 22th August
 Date 22th August 2023
 Postscript: An individual program to generate maps. Including the function of the DataFrameWriter.py
"""
import math
import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd
from math import log10,sin,pow,sqrt
from queue import Queue
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib.patches import RegularPolygon
import matplotlib.patches as patches
from sklearn.metrics import silhouette_score,silhouette_samples
from sklearn.neighbors import KernelDensity
from scipy.interpolate import griddata
from sklearn.cluster import KMeans


def kmeans(X, K, max_iters=300):
    centroids = initialize_centroids(X, K)
    for _ in range(max_iters):
        labels = assign_samples_to_centroids(X, centroids)
        new_centroids = update_centroids(X, labels, K)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

def initialize_centroids(X, K):
    random_indices = random.sample(range(len(X)), K)
    centroids = X[random_indices]
    return centroids

# Calculate the distance between two latitude and longitude coordinates (unit: meters)
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    distance = 6371 * c * 1000  # The radius of the earth is 6371 kilometers, multiplied by 1000 to convert to meters
    return distance

# Find the midpoints of a regular hexagon and other adjacent regular hexagons
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

def assign_samples_to_centroids(X, centroids):
    labels = []
    for sample in X:
        distances = [np.linalg.norm(sample - centroid) for centroid in centroids]
        label = np.argmin(distances)
        labels.append(label)
    return labels

def update_centroids(X, labels, K):
    new_centroids = np.zeros((K, X.shape[1]))
    counts = np.zeros(K)
    for i, sample in enumerate(X):
        label = labels[i]
        new_centroids[label] += sample
        counts[label] += 1
    for j in range(K):
        if counts[j] > 0:
            new_centroids[j] /= counts[j]
    return new_centroids

# Calculate path loss through cost231-Hata model
def cost231_hata_model( distance ):
    frequency = 2100
    hb = 30
    hr = 1.5
    alpha = (1.1 * math.log10(frequency) - 0.7) * hr - (1.56 * math.log10(frequency) - 0.8)
    path_loss = 46.3 + 33.9 * math.log10(frequency) - 13.82 * math.log10(hb) - alpha + (
                44.9 - 6.55 * math.log10(hb)) * math.log10(distance)
    return path_loss

# RMa is suitable for rural areas, including LOS (Line of Sight) and NLOS (Line of Sight), which requires the height of the base station to be 5 to 150m, the height of the mobile phone to be 1 to 10m, and the height of the building to be 5 to 50m.
# Default value (expected to be the most applicable value): base station height 35m, mobile phone height 1.5m, building height 5m.
def RMa_LOS_d2d(d2d):
    # parameter
    # Ground distance, tower base to UE
    fc=0.5 # Unit in GHz.Frenquence center See the top line of the protocol 7.4.1 table and the note6 below
    Hbs=21 # base station height
    Hue=1.5 # UE height
    Hbu=8  # building  height
    d3d=sqrt(d2d**2+(Hbs-Hue)**2)   # The spatial distance, from the base station antenna to the UE antenna, is calculated through the Pythagorean theorem.
    # dbp calculating
    # Use () for the denominator, otherwise divide the load and multiply it by the power first, which will cause an error.
    dbp=2*3.1415926*Hbs*Hue*fc*10**9/(3*10**8)
    PL1=20*log10(40*3.1415926*d3d*fc/3)+min(0.03*pow(Hbu,1.72),10)*log10(d3d)\
    -min(0.044*pow(Hbu,1.72),14.77)+0.002*log10(Hbu)*d3d
    PL2=PL1+40*log10(d3d/dbp)
    # path loss calculating
    if d2d<=dbp:
        PL=PL1
    if d2d>dbp:
        PL=PL2
    CL = log10(4 + 2 * fc)
    return PL+CL

def calculate_SBS_received_power(transmit_power, distance):
    # Unit: dBm
    transmit_power_dbm = 10 * math.log10(transmit_power)
    # Unit: dB
    path_loss = RMa_LOS_d2d(distance)
    # Unit: dBm
    received_power_dbm = transmit_power_dbm - path_loss
    return received_power_dbm

def calculate_MBS_received_power(transmit_power, distance):
    # Unit: dBm
    transmit_power_dbm = 10 * math.log10(transmit_power)
    # Unit: dB
    path_loss = cost231_hata_model(distance)
    # Unit: dBm
    received_power_dbm = transmit_power_dbm - path_loss
    return received_power_dbm

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Determine whether a center point is a high-traffic density point
def is_high_traffic_center(center_x, center_y, points, radius):
    count = 0
    for point in points:
        distance = euclidean_distance(center_x, center_y, point[0], point[1])
        if distance <= radius:
            count += 1
    return count

data1 = pd.read_csv('./data/Los_Angeles_136k.csv')[0:6000]
data=np.array(data1)
testdata=np.array(data1)

# Calculate the limit value of the coordinates represented by latitude and longitude, that is, the range that the base station can select (-0.01 or +0.01 to slightly expand the range)
lat_min, lat_max = data[:,1].min()-0.01, data[:,1].max()+0.01
lon_min, lon_max = data[:,2].min()-0.01, data[:,2].max()+0.01

USER=[]
for i in range(len(data)):
    USER.append([data[i][1],data[i][2]])

testUSER=[]
for i in range(len(testdata)):
    testUSER.append([testdata[i][1],testdata[i][2]])

# rectangle dimensions
latDelta = lat_max - lat_min
lonDelta = lon_max - lon_min
areaTotal = latDelta * lonDelta  # area of rectangle
lam = 30

# MBS deployment
MBS_R_km = 2  # coverage radius
MBS_R = MBS_R_km/111  # Scale conversion from km to latitude and longitude
center_point=(round((lat_max-lat_min)/2+lat_min,3),round((lon_max-lon_min)/2+lon_min,3))

# The midpoint of the initial regular hexagon
MBS_coordinates=[center_point]
adjacent_hexagons = find_adjacent_hexagons(center_point[0], center_point[1], MBS_R)

# Find all regular hexagons whose center points are within the area
queue_hexagons = Queue()
for point in adjacent_hexagons:
    queue_hexagons.put(point)
while not queue_hexagons.empty():
    point=queue_hexagons.get()
    if is_within_range(point,lat_min, lat_max,lon_min, lon_max) and is_not_in_center_coordinates(point,MBS_coordinates):
            # print(point)
            MBS_coordinates.append(point)
            adjacent_hexagons=find_adjacent_hexagons(point[0],point[1],MBS_R)
            for point in adjacent_hexagons:
                if is_not_in_center_coordinates(point,MBS_coordinates) and is_within_range(point, lat_min, lat_max,lon_min, lon_max):
                        queue_hexagons.put(point)

# print(MBS_coordinates)
# Draw a diagram showing the distribution of macro base stations and users. Red represents users and blue represents base stations.
fig, ax = plt.subplots()
plt.scatter(testdata[:,1],testdata[:,2], c='r', marker='o',s=5)

# Create a rectangle patch
rect = patches.Rectangle((lat_min, lon_min), latDelta, lonDelta, linewidth=1, edgecolor='r', facecolor='none')

# MBS
for center_x, center_y in MBS_coordinates:
    # Create the regular hexagon centered at (center_x, center_y)

    hexagon = RegularPolygon((center_x, center_y), numVertices=6, radius=MBS_R, edgecolor='black',
                             facecolor=(0.6, 0.6, 0.6, 0.01), orientation=np.radians(30), alpha=0.1)
    plt.scatter(center_x, center_y, color='black', marker='^', s=10, alpha=0.6)
    # Add the hexagon to the plot
    ax.add_patch(hexagon)

# Add the rectangle to the plot
ax.add_patch(rect)

# Set the axis limits based on the maximum and minimum coordinates
x_coordinates, y_coordinates = zip(*MBS_coordinates)
plt.xlim(min(x_coordinates) - MBS_R * 1.5, max(x_coordinates) + MBS_R * 1.5)
plt.ylim(min(y_coordinates) - MBS_R * 1.5, max(y_coordinates) + MBS_R * 1.5)

# Set axis labels
plt.xlabel('X')
plt.ylabel('Y')

# Show the plot
plt.axis('equal')
plt.grid(True)
plt.title('Traffic demand distribution based on K-means')
plt.show()


# Calculate the macro base station connected to each user (the macro base station with the strongest signal)
SIG=[]
MBS_USER=[]
for j in range(len(MBS_coordinates)):
    MBS_USER.append([])
transmit_power=30
for i in range(len(testUSER)):
    signal_intensity=-500
    num=0
    for j in range(len(MBS_coordinates)):
        distance=abs(haversine_distance(testUSER[i][0], testUSER[i][1], MBS_coordinates[j][0], MBS_coordinates[j][1]))/1000
        if signal_intensity<calculate_MBS_received_power(transmit_power, distance):
            signal_intensity=calculate_MBS_received_power(transmit_power, distance)
            num=j
    MBS_USER[num].append(testUSER[i])
    SIG.append(signal_intensity)
# print(SIG)

signal_power_interval = {'135':0,'125':0,'115': 0, '105': 0, '95': 0, '85': 0, '75': 0, '65': 0, '55': 0, '45': 0, '35': 0}
res = []
for sig in SIG:
    for key in signal_power_interval.keys():
        if sig<=0-int(key):
            signal_power_interval[key]=signal_power_interval[key]+1
            break

for key,value in signal_power_interval.items():
    res.append(value/len(SIG))
    #print(0-int(key),' ',value/len(SIG))
print(res)


USER=np.array(USER)
# Using KDE to estimate traffic demand density
kde = KernelDensity(kernel='gaussian', bandwidth=0.20).fit(USER)  # 0.02
log_dens = kde.score_samples(USER)
# Selecting the top dense points as initial centroids
initial_k = 268
distance_limit_by_SBS_R = 0.69/111
# Implementing the approach to get 14 unique dense points
n = 1
unique_centroids = []
unique_centroids_count = 0
while unique_centroids_count < initial_k:
    z = n * initial_k
    initial_centroids_indices = log_dens.argsort()[-z:][::-1]
    initial_centroids = USER[initial_centroids_indices]
    unique_centroids = np.unique(initial_centroids, axis=0) # Find unique rows in the numpy array and count them
    unique_centroids_count = len(unique_centroids)
    n += 1

# Filter the centroids by euclidean_distance greater than SBS_R
filtered_centroids = []
for centroid in unique_centroids:
    x, y = centroid
    valid = all(euclidean_distance(x, y, c[0], c[1]) > distance_limit_by_SBS_R for c in filtered_centroids)
    if valid:
        filtered_centroids.append(centroid)
k = len(filtered_centroids)
print('k值：', k)
# Once we have at least 14 unique points, select the top 14
final_centroids = filtered_centroids[:k]

# Find hot spots and deploy SBS based on Kmeans
SBS_num=13
SBS_R_km=0.5
SBS_R=SBS_R_km/111

Kmeans = KMeans(n_clusters=k, init=final_centroids, random_state=1, n_init=1).fit(USER)
labels, centroids = Kmeans.labels_,Kmeans.cluster_centers_
# print(centroids)

dist_centroids={}
for center_x,center_y in centroids:
    dist_centroids[(center_x, center_y)]=[]

radius = SBS_R
min_count = 20
SBS_centroids=[]
del_centroids=[]
hot_points=[]

for center_x, center_y in centroids:
    point_density=is_high_traffic_center(center_x, center_y, USER, radius)
    dist_centroids[(center_x, center_y)].append(point_density)
    if point_density > min_count:
        hot_points.append([center_x, center_y])
    else:
        del_centroids.append((center_x, center_y))

silhouette_scores = silhouette_score(USER, labels)
print(silhouette_scores)
silhouette_sample=silhouette_samples (USER, labels)
cluster_scores = []
for cluster_label in range(k):
    cluster=[]
    for i in range(len(USER)):
        if labels[i]==cluster_label:
            cluster.append(silhouette_sample[i])
    cluster=np.array(cluster)
    mean_cluster_silhouette_score = np.mean(cluster)
    # cluster_scores.append(mean_cluster_silhouette_score)
    dist_centroids[(centroids[cluster_label][0], centroids[cluster_label][1])].append(mean_cluster_silhouette_score)
print(cluster_scores)

centroids_to_centroids=np.zeros(k)
for i in range(k):
    if centroids[i][0]==0 and centroids[i][1]==0:
        continue
    dis=0
    for j in range(k):
        if centroids[i][0] == 0 and centroids[i][1] == 0:
            continue
        dis=dis+euclidean_distance(centroids[i][0],centroids[i][1],centroids[j][0],centroids[j][1])
    centroids_to_centroids[i]=dis
    dist_centroids[(centroids[i][0], centroids[i][1])].append(dis)
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
    print('Amount of Fill-in Coverage Candidates is lesser than SBS_num:', len(SBS_ranking_all_parameter_output))

# Save as DataFrame
SBS_centroids_DF=pd.DataFrame(SBS_ranking_all_parameter_output,
                              columns=['Latitude', 'Longitude', 'Point Density', 'Mean Silhouette'])
SBS_centroids_DF.to_csv('Fill-in_Coverage_Candidates_based_on_Kmeans.csv', index=False)

transmit_power=10  # Unit: W
Kmeans_SIG=[]
for i in range(len(testUSER)):
    signal_intensity=SIG[i]
    for j in range(len(SBS_centroids)):
        distance=abs(haversine_distance(testUSER[i][0], testUSER[i][1], SBS_centroids[j][0], SBS_centroids[j][1]))
        if signal_intensity<calculate_SBS_received_power(transmit_power, distance):
            signal_intensity=calculate_SBS_received_power(transmit_power, distance)
    Kmeans_SIG.append(signal_intensity)

signal_power_interval = {'135':0,'125':0,'115': 0, '105': 0, '95': 0, '85': 0, '75': 0, '65': 0, '55': 0, '45': 0, '35': 0}
res = []
for sig in Kmeans_SIG:
    for key in signal_power_interval.keys():
        if sig<=0-int(key):
            signal_power_interval[key]=signal_power_interval[key]+1
            break

for key,value in signal_power_interval.items():
    res.append(value/len(SIG))
    #print(0-int(key),' ',value/len(SIG))
Kmeans_SIG=np.array(Kmeans_SIG)
print(np.mean(Kmeans_SIG))
print(res)

SBS_random_centroids=centroids[random.sample(range(1, k), SBS_num)]
transmit_power=10
Kmeans_SIG = []
for i in range(len(testUSER)):
    signal_intensity=SIG[i]
    for j in range(len(SBS_random_centroids)):
        distance=abs(haversine_distance(testUSER[i][0], testUSER[i][1], SBS_random_centroids[j][0], SBS_random_centroids[j][1]))
        if signal_intensity<calculate_SBS_received_power(transmit_power, distance):
            signal_intensity=calculate_SBS_received_power(transmit_power, distance)
    Kmeans_SIG.append(signal_intensity)
#print(Kmeans_SIG)

signal_power_interval = {'135':0,'125':0,'115': 0, '105': 0, '95': 0, '85': 0, '75': 0, '65': 0, '55': 0, '45': 0, '35': 0}
res=[]
for sig in Kmeans_SIG:
    for key in signal_power_interval.keys():
        if sig<=0-int(key):
            signal_power_interval[key]=signal_power_interval[key]+1
            break

for key,value in signal_power_interval.items():
    res.append(value/len(SIG))
    #print(0-int(key),' ',value/len(SIG))

print(res)
Kmeans_SIG=np.array(Kmeans_SIG)
print(np.mean(Kmeans_SIG))

# Plot clustering results
# Draw picture Macro base station and user distribution map, red represents users, blue represents base stations
fig, ax = plt.subplots(figsize=(3.5, 3.5 * lonDelta / latDelta))  # Canvas of IEEE spec picture with width 3.5 inches
colors = [ 'b', 'g', 'r', 'c', 'm', 'y']
for i, point in enumerate(testUSER):
    cluster = labels[i]
    color = colors[cluster % len(colors)]
    plt.scatter(point[0], point[1], c=color, marker='o',s=5)

# Create a rectangle patch
rect = patches.Rectangle((lat_min, lon_min), latDelta, lonDelta, linewidth=1, edgecolor='r', facecolor='none', alpha=0.6)
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
    circle = plt.Circle((center_x, center_y), SBS_R, color='r', fill=False)
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
# plt.title('Identified hot spots by scoring and ranking based on K-means')
plt.savefig('Identified fill-in coverage candidates based on Kmeans.png', dpi=300, bbox_inches='tight')
plt.show()