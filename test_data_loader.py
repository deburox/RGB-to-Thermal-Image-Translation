
from training.data_loader import load_datasets

print("Loading reduced datasets...")

train_dataset, test_dataset = load_datasets()

print("\nDataset loading successful!")

for rgb, thermal in train_dataset.take(1):
    print("\nTraining batch:")
    print("RGB shape:", rgb.shape)
    print("Thermal shape:", thermal.shape)

for rgb, thermal in test_dataset.take(1):
    print("\nTesting batch:")
    print("RGB shape:", rgb.shape)
    print("Thermal shape:", thermal.shape)

print("\nData loader test completed!")
