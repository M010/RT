import numpy as np
import matplotlib.pyplot as plt


def normalize(vector):
    return vector / np.linalg.norm(vector)


def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis


def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None


def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


def get_illumination(objects, nearest_object, normal_to_surface, shifted_point):
    illumination = np.zeros((3))  # black
    intersection_to_light = normalize(light['position'] - shifted_point)

    _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
    intersection_to_light_distance = np.linalg.norm(light['position'] - shifted_point)
    is_shadowed = min_distance < intersection_to_light_distance

    if is_shadowed:
        return illumination

    ambient = nearest_object['ambient'] * light['ambient']
    diffuse = nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light,
                                                                    normal_to_surface)
    intersection_to_camera = normalize(camera - shifted_point)
    H = normalize(intersection_to_light + intersection_to_camera)
    specular = nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (
            nearest_object['shininess'] / 4)

    return ambient + diffuse + specular


def trace_ray(objects, origin, ray_direction, depth):
    nearest_object, min_distance = nearest_intersected_object(objects, origin, ray_direction)
    if nearest_object is None:
        return np.zeros(3)
    intersection = origin + min_distance * ray_direction
    normal_to_surface = normalize(intersection - nearest_object['center'])
    shifted_point = intersection + 1e-5 * normal_to_surface
    color = get_illumination(objects, nearest_object, normal_to_surface, shifted_point)
    # reflection
    origin = shifted_point
    ray_direction = reflected(ray_direction, normal_to_surface)
    if depth > 0:
        color += trace_ray(objects, origin, ray_direction, depth - 1) * nearest_object['reflection']
    return np.clip(color, 0, 1)


width = 500
height = 500
max_depth = 3

camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

light = {'position': np.array([5, 10, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]),
         'specular': np.array([1, 1, 1])}

objects = [
    {'center': np.array([-0.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]),
     'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5},
    {'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0.6, 0.1]),
     'diffuse': np.array([0.7, 0.5, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5},
    {'center': np.array([-0.3, 0, 0]), 'radius': 0.30, 'ambient': np.array([0, 0.1, 0]),
     'diffuse': np.array([0, 0.6, 0.9]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.7},
    {'center': np.array([0, -900, 0]), 'radius': 899, 'ambient': np.array([0.1, 0.1, 0.1]),
     'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5}
]

image = np.zeros((height, width, 3))
size = 0
print("Process, please wait:")
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        # screen is on origin
        pixel = np.array([x, y, 0])
        direction = normalize(pixel - camera)
        color = trace_ray(objects, camera, direction, 3)
        image[i, j] = color

    tmp_str = '{}/100'.format(int((i + 1) * 100 / height))
    print("\r" * size, end="")
    size = len(tmp_str)
    print(tmp_str, end="")

plt.imsave('image.png', image)
