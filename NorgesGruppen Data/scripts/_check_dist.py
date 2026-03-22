import json, collections, sys

with open("artifacts/crop_classifier_manifest.json") as f:
    m = json.load(f)

train_counts = collections.Counter(s["category_id"] for s in m["train_samples"])
val_counts   = collections.Counter(s["category_id"] for s in m["val_samples"])
counts = sorted(train_counts.items(), key=lambda x: x[1])

print(f"Total classes in train : {len(train_counts)}")
print(f"Min  samples: {counts[0]}")
print(f"Max  samples: {counts[-1]}")
print(f"Median       : {counts[len(counts)//2]}")
print()
print("Bottom 30 classes (fewest train samples):")
for cid, cnt in counts[:30]:
    print(f"  class {cid:3d}: {cnt:4d} train  {val_counts.get(cid, 0):3d} val")
print()
print(f"Classes <=  5 train: {sum(1 for _,c in counts if c <=  5)}")
print(f"Classes <= 10 train: {sum(1 for _,c in counts if c <= 10)}")
print(f"Classes <= 20 train: {sum(1 for _,c in counts if c <= 20)}")
print(f"Classes <= 50 train: {sum(1 for _,c in counts if c <= 50)}")
print()

# Sample entry structure
print("Sample train entry:")
print(m["train_samples"][0])

# Check source types
sources = collections.Counter(s.get("source", "shelf") for s in m["train_samples"])
print()
print("Source breakdown:", dict(sources))
