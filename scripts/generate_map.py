import re
import json

with open("gallery.md") as f:
    content = f.read()


pattern = r"([a-zA-Z]+(?:_[a-zA-Z]+)*)\s\((\d+?)\).*`(n\d+)`"
ans = dict()
for line in content.split("|"):
    print(line)
    m = re.search(pattern, line)
    if m is not None:
        c_name, c_ix, img_id = m.groups()
        ans[img_id] = (c_ix, c_name)

with open("imagenet_map_classes.json", "w") as f:
    json.dump(ans, f)
