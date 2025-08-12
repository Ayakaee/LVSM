import objaverse
import os
# 获取所有对象的UID
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
uids = objaverse.load_uids()
print(f"Total objects: {len(uids)}")