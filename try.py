with open('data/test/full_list.txt', 'r') as f:
    all_scene_paths = f.read().splitlines()
print(len(all_scene_paths))