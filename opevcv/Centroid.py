def centroid (*points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    print(x_coords, y_coords)
    centroid_x = sum(x_coords)/len(points)
    centroid_y = sum(y_coords)/len(points)
    return [centroid_x, centroid_y]