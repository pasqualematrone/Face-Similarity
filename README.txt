Python tool that uses a convolutional neural network (ResNet-50) to quantify, through the attribution of a score, the difference between the same face wearing or not wearing a health mask.

Tells the similarity score for the 2 face images (face matching).

It taked 2 image files as input then it extract the face from it and process the image file.

After that it extract face features encoding from it and calculate the cosine distance between 2 files.

Threshold value of 0.5 is set and then it compares it with that value to tell whether it's a match or not.

Use face_array() function if the 2 input images are already faces, use extract_face() function otherwise.
