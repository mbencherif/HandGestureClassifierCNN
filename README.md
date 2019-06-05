# Hand Gesture Classifier
CNN approach to video level hand gesture classification

With the constant improvements in performance of modern processors, fueling the rising popularity of machine learning, many opportunities present themselves in the area of human-computer interfaces. One such possibility is the ability for computers to utilize a cheap webcam to recognize hand gestures.

# Approach
When it comes to image and video processing, the convolutional neural network shines among other neural network architectures. However, the biggest challenge in this project comes from the requirement for low latency processing.

The current approach is to fuse optical flow data with image data and feed it into a single frame model, for which the outputs are then fed into a LSTM. Further work will be done in exploring 3D convolutions and attention structures.

The model is currently being trained on the Jester dataset: https://20bn.com/datasets/jester
To run the training task, make sure that the `base_location` variable is set to the home directory of where the data set is located.
