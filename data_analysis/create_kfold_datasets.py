import os
import shutil
from sklearn.model_selection import KFold

# Specify the path to the original UCMerced Landuse dataset directory
original_dataset_dir = "./../research_container/container/src/data/UCMerced_Landuse"

# Specify the base directory where the fold datasets will be created
base_output_dir = "./../research_container/container/src/data/UCMerced_Landuse_5Folds"

# Create the base output directory if it doesn't exist
os.makedirs(base_output_dir, exist_ok=True)

# Get the list of image filenames and their corresponding labels
classes = sorted(os.listdir(original_dataset_dir))
classes = classes[1:]

print(len(classes))
num_classes = len(classes)

image_files = []
labels = []
for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(original_dataset_dir, class_name)
    class_images = sorted(os.listdir(class_dir))
    image_files.extend([os.path.join(class_name, img) for img in class_images])
    labels.extend([class_idx] * len(class_images))

# Create a KFold object with 5 splits and shuffle the data
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over the folds and create new dataset directories
for fold, (train_idx, test_idx) in enumerate(kfold.split(image_files)):
    fold_dir = os.path.join(base_output_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # Create train and test directories for the current fold
    train_dir = os.path.join(fold_dir, "train")
    test_dir = os.path.join(fold_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy train images to the train directory
    for idx in train_idx:
        src_path = os.path.join(original_dataset_dir, image_files[idx])
        dst_path = os.path.join(train_dir, image_files[idx])
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copyfile(src_path, dst_path)

    # Copy test images to the test directory
    for idx in test_idx:
        src_path = os.path.join(original_dataset_dir, image_files[idx])
        dst_path = os.path.join(test_dir, image_files[idx])
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copyfile(src_path, dst_path)

    print(f"Created dataset directories for fold {fold}.")