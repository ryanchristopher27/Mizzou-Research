import os
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Specify the directory where you want to store the CIFAR-10 data
data_dir = "./../research_container/container/src/data/CIFAR10"

# Create the data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and extract the CIFAR-10 train dataset
train_dataset = datasets.CIFAR10(
    root=data_dir,
    train=True,
    download=True,
    transform=transform
)

# Download and extract the CIFAR-10 test dataset
test_dataset = datasets.CIFAR10(
    root=data_dir,
    train=False,
    download=True,
    transform=transform
)

# Create the train and test directories
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Save the train images to the train directory
for i in range(len(train_dataset)):
    image, label = train_dataset[i]
    class_dir = os.path.join(train_dir, str(label))
    os.makedirs(class_dir, exist_ok=True)
    image_path = os.path.join(class_dir, f"image_{i}.png")
    torchvision.utils.save_image(image, image_path)

# Save the test images to the test directory
for i in range(len(test_dataset)):
    image, label = test_dataset[i]
    class_dir = os.path.join(test_dir, str(label))
    os.makedirs(class_dir, exist_ok=True)
    image_path = os.path.join(class_dir, f"image_{i}.png")
    torchvision.utils.save_image(image, image_path)

print("CIFAR-10 data saved to:", data_dir)