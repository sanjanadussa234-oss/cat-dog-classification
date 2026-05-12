import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Dataset Paths
train_dir = "dataset/train"
val_dir = "dataset/validation"
test_dir = "dataset/test"

# Get Classes
classes = os.listdir(train_dir)

print("\nClasses Found:")
print(classes)

# -----------------------------------
# COUNT IMAGES PER CLASS
# -----------------------------------

image_counts = {}

for cls in classes:
    cls_path = os.path.join(train_dir, cls)
    image_counts[cls] = len(os.listdir(cls_path))

print("\nImage Count Per Class:")
print(image_counts)

# -----------------------------------
# VISUALIZE CLASS DISTRIBUTION
# -----------------------------------

plt.figure(figsize=(6, 4))

sns.barplot(
    x=list(image_counts.keys()),
    y=list(image_counts.values())
)

plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Images")

# Save Graph
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/class_distribution.png")

plt.show()

# -----------------------------------
# DISPLAY SAMPLE IMAGES
# -----------------------------------

plt.figure(figsize=(10, 5))

for i, cls in enumerate(classes):

    img_name = os.listdir(os.path.join(train_dir, cls))[0]

    img_path = os.path.join(train_dir, cls, img_name)

    img = Image.open(img_path)

    plt.subplot(1, 2, i + 1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.savefig("reports/sample_images.png")

plt.show()

# -----------------------------------
# CHECK IMAGE DIMENSIONS
# -----------------------------------

sizes = []

for cls in classes:

    cls_path = os.path.join(train_dir, cls)

    for img_name in os.listdir(cls_path)[:50]:

        img_path = os.path.join(cls_path, img_name)

        img = Image.open(img_path)

        sizes.append(img.size)

print("\nSample Image Sizes:")
print(sizes[:10])

# -----------------------------------
# CHECK CORRUPTED IMAGES
# -----------------------------------

corrupted = []

for cls in classes:

    cls_path = os.path.join(train_dir, cls)

    for img_name in os.listdir(cls_path):

        img_path = os.path.join(cls_path, img_name)

        try:
            img = Image.open(img_path)
            img.verify()

        except:
            corrupted.append(img_path)

print("\nCorrupted Images Found:", len(corrupted))

if len(corrupted) > 0:

    print("\nCorrupted Files:")

    for file in corrupted:
        print(file)

print("\nEDA Completed Successfully!")