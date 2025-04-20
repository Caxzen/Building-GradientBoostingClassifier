import numpy as np
import os

def generate_dataset(n_samples=1000, random_seed=42):
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Generate two features
    feature1 = np.random.randn(n_samples)
    feature2 = np.random.randn(n_samples)
    
    # Create labels: 1 if feature1 + feature2 > 0 (with some noise), else 0
    noise = np.random.randn(n_samples) * 0.2
    labels = (feature1 + feature2 + noise > 0).astype(int)
    
    # Combine into a single array
    data = np.column_stack((feature1, feature2, labels))
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV with header
    header = 'feature1,feature2,label'
    np.savetxt('data/dataset.csv', data, delimiter=',', header=header, fmt='%.6f', comments='')
    print(f"Dataset generated and saved to 'data/dataset.csv' with {n_samples} samples.")

if __name__ == "__main__":
    generate_dataset()