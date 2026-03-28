import tensorflow as tf
from PIL import Image
import numpy as np

# Load real MNIST data
(x_test, y_test), _ = tf.keras.datasets.mnist.load_data()

# Save one image per digit (0-9)
for digit in range(10):
    # Find first occurrence of each digit
    idx = np.where(y_test == digit)[0][0]
    img_array = x_test[idx]  # already 28x28, black bg white digit
    
   # New — keeps original MNIST format
    img = Image.fromarray(img_array)
    img.save(f"test_digit_{digit}.png")

print("Saved 10 test images: test_digit_0.png ... test_digit_9.png")