import osmnx as ox
import networkx as nx
import folium
import numpy as np
import matplotlib.pyplot as plt

# Download and prepare the graph
place_name = "Dublin City, Ireland"
G = ox.graph.graph_from_place(place_name, network_type="drive")

#G = ox.simplification.consolidate_intersections(
#        G,
#        tolerance=0.0002,
#        rebuild_graph=True,
#        dead_ends=False 
#    )

# Convert graph to GeoDataFrames
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

print(f'Node info: {nodes_gdf.info()}')
print(f'Edge info: {edges_gdf.info()}')

A = nx.to_numpy_array(G, weight="length")  # or weight="lanes"
np.save("adj_matrix", A)


# Calculate map center
center_lat = (nodes_gdf.y.min() + nodes_gdf.y.max()) / 2
center_lon = (nodes_gdf.x.min() + nodes_gdf.x.max()) / 2

# Create the map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles='OpenStreetMap'
)

# Add edges to the map
for idx, row in edges_gdf.iterrows():
    if row.geometry is not None:

        # Convert coordinates for Folium [lat, lon] format
        coords = [[point[1], point[0]] for point in row.geometry.coords]
        
        # Create popup with street information
        popup_text = f"Street: {row.get('name', 'Unnamed')}<br>Type: {row.get('highway', 'Unknown')}<br>Reversed: {row.get('reversed', 'Unknown')}"


        if idx[0] < 20:
            print(f'Num lanes : {row.lanes}')
            print(f'Type : {row.highway}')
            try:
                folium.PolyLine(
            locations=coords,
            color='orange',
            weight=2,
            opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200)
        ).add_to(m)
            except:        
                folium.PolyLine(
                    locations=coords,
                    color='blue',
                    weight=2,
                    opacity=0.7,
                    popup=folium.Popup(popup_text, max_width=200)
                ).add_to(m)

        else:
            folium.PolyLine(
            locations=coords,
            color='blue',
            weight=2,
            opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200)
        ).add_to(m)

# Add some major intersections as markers
major_intersections = [node for node, degree in G.degree() if degree >= 8]
for node_id in major_intersections:
    node_data = nodes_gdf.loc[node_id]
    folium.CircleMarker(
        location=[node_data.y, node_data.x],
        radius=4,
        popup=f"Major Intersection<br>Node: {node_id}<br>Connections: {G.degree(node_id)}",
        color='red',
        fillColor='red',
        fillOpacity=0.8
    ).add_to(m)


nodes_gdf.to_csv("node_data.csv")
edges_gdf.to_csv("edges_data.csv")

# Save the map
m.save("dublin_interactive_osmnx.html")
print("Interactive map saved as 'dublin_interactive_osmnx.html'")

# Display basic graph info
print(f"Graph contains {len(G.nodes)} nodes and {len(G.edges)} edges")
# print(f"Found {len(major_intersections)} major intersections (degree >= 4)")

degrees = np.sum(A, axis=1)
D = np.diag(degrees)
L = D - A
eigvals, eigvecs = np.linalg.eigh(L)
#eigenvalues (connectivity, bottlenecks),eigenvectors (natural partitions / “veins” of travel)

# pick second eigenvector
fiedler = eigvecs[:,1]

plt.figure(figsize=(10,8))
nx.draw(
    G, 
    pos={n:(nodes_gdf.loc[n].x, nodes_gdf.loc[n].y) for n in G.nodes()}, 
    node_color=fiedler, 
    cmap=plt.cm.viridis, 
    node_size=10, 
    edge_color="lightgrey"
)
plt.show()

