# Source: https://github.com/abhay6694/Face-Similarity
# Tells the similarity score for the 2 images based on their face matching.
# It taked 2 image files as input then it extract the face from it and process the image file.
# After that it extract face features encoding from it and calculate the cosine distance between 2 files.
# Threshold value of 0.5 is set and then it compares it with that value to tell whether it's a match or not.

# Install the following libraries:
# pip install tensorflow
# pip install keras
# pip install pillow
# pip install scipy
# pip install mtcnn
# pip install matplotlib
# pip install keras_applications

# Network Model: resnet50
# Face Detector: MTCNN
# Measure of Similarity: cosine similarity to measure the similarity between embeddings

# *** Face Similarity ***

# face verification with the VGGFace2 model
import PIL
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from utils import preprocess_input
from vggface import VGGFace
import sys
import cv2
import tensorflow as tf
import pathlib
from os import listdir
from os.path import isfile, join
import os

masked_non_detected_counter = 0
unmasked_non_detected_counter = 0


# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    global masked_non_detected_counter
    global unmasked_non_detected_counter

    img = plt.imread(filename)
    detector = MTCNN()
    results = detector.detect_faces(img)
    try:
        x1, y1, width, height = results[0]['box']
    except Exception as e:
        print(filename, "→ Face Not Detected:", e)

        if filename.startswith('Masked_'):
            masked_non_detected_counter += 1
        else:
            unmasked_non_detected_counter += 1
        # cv2.imshow(filename, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        image = Image.fromarray(img)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        return face_array
        # exit()
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = img[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)

    # Drawing a rectangle around the face detected
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    # cv2.imshow(filename, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return face_array

def face_array(filename, required_size=(224, 224)):
    global masked_non_detected_counter
    global unmasked_non_detected_counter

    img = plt.imread(filename)

    # Gaussian filter kernel 5x5
    # img = cv2.GaussianBlur(img, (5,5),0)

    masked_non_detected_counter = "Disabled"
    unmasked_non_detected_counter = "Disabled"

    # resize pixels to the model size
    image = Image.fromarray(img)
    image = image.resize(required_size)
    face_array = np.asarray(image)

    return face_array


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
    # extract faces
    # faces = [extract_face(f) for f in filenames]
    # no extract faces
    faces = [face_array(f) for f in filenames]
    # convert into an array of samples
    samples = np.asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)

    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg') # model='vgg16'
    # model.summary() # View all the layers of the network using the model's Model.summary method
    pred = model.predict(samples)
    return pred


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)

    # print(known_embedding)
    # print(candidate_embedding)
    # print(score)

    print('*******************************************************************')
    print('Threshold for the face similarity score is 0.5')

    if score <= thresh:
        print('Face is a Match with score of %.3f' % score)
    else:
        print('Face is not a Match with score of %.3f' % score)

    print('*******************************************************************')

    return score


def main():
    print("Starting...")

    score_same_person = []
    score_different_person = []
    files = [f for f in listdir("Test/Masked and Unmasked/Masked") if isfile(join("Test/Masked and Unmasked/Masked", f))]

    for i in files:
        img1 = cv2.imread("Test/Masked and Unmasked/Masked/" + i)
        res1 = cv2.resize(img1, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite("Masked_" + i, res1)
        print("Masked_" + i)

        img2 = cv2.imread("Test/Masked and Unmasked/UnMasked/" + i)
        res2 = cv2.resize(img2, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite("UnMasked_" + i, res2)
        print("UnMasked_" + i)

        embeddings = get_embeddings(["Masked_" + i, "UnMasked_" + i])
        score_same_person.append(is_match(embeddings[0], embeddings[1]))

        os.remove("Masked_" + i)
        os.remove("UnMasked_" + i)

    len_file = len(files)
    for i in range(len_file):
        if i + 1 <= len_file - 1:
            img1 = cv2.imread("Test/Masked and Unmasked/Masked/" + files[i])
            res1 = cv2.resize(img1, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imwrite("Masked_" + files[i], res1)
            print("Masked_" + files[i])

            img2 = cv2.imread("Test/Masked and Unmasked/UnMasked/" + files[i + 1])
            res2 = cv2.resize(img2, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imwrite("UnMasked_" + files[i + 1], res2)
            print("UnMasked_" + files[i + 1])

            embeddings = get_embeddings(["Masked_" + files[i], "UnMasked_" + files[i + 1]])
            score_different_person.append(is_match(embeddings[0], embeddings[1]))

            os.remove("Masked_" + files[i])
            os.remove("UnMasked_" + files[i + 1])
        else:
            img1 = cv2.imread("Test/Masked and Unmasked/Masked/" + files[i])
            res1 = cv2.resize(img1, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imwrite("Masked_" + files[i], res1)
            print("Masked_" + files[i])

            img2 = cv2.imread("Test/Masked and Unmasked/UnMasked/" + files[0])
            res2 = cv2.resize(img2, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imwrite("UnMasked_" + files[0], res2)
            print("UnMasked_" + files[0])

            embeddings = get_embeddings(["Masked_" + files[i], "UnMasked_" + files[0]])
            score_different_person.append(is_match(embeddings[0], embeddings[1]))

            os.remove("Masked_" + files[i])
            os.remove("UnMasked_" + files[0])

    print("*******************************************************************")
    print("Score Same People →", score_same_person)
    print("*******************************************************************")
    print("Score Different People →", score_different_person)

    y = score_same_person
    y2 = score_different_person
    x = np.arange(1, len(score_same_person)+1, dtype=int)
    # Plot
    plt.scatter(x, y, c='g')
    plt.scatter(x, y2, c='r')
    # Add Title
    plt.title("Face Similarity")
    # Add Axes Labels
    plt.xlabel("Iteration number")
    plt.ylabel("Threshold value")
    # Plot Threshold
    plt.axhline(y=0.5)
    # Display
    plt.show()

    # Calculate the accuracy
    results_same_person = []
    results_different_person = []

    for i in range(len(score_same_person)):
        if(score_same_person[i] <= 0.5):
            results_same_person.append(0)
        else:
            results_same_person.append(1)

    for i in range(len(score_different_person)):
        if(score_different_person[i] > 0.5):
            results_different_person.append(0)
        else:
            results_different_person.append(1)

    print(results_same_person)
    print(results_different_person)

    # Metriche di valutazione delle prestazioni
    correct_same_person = len(results_same_person) - np.count_nonzero(results_same_person)
    correct_different_person = len(results_different_person) -np.count_nonzero(results_different_person)

    true_positive = correct_same_person
    false_positive = len(results_same_person) - true_positive
    true_negative = correct_different_person
    false_negative = len(results_different_person) - true_negative
    print("True positive:", true_positive) # Stessa persona (con e senza mascherina) classificata correttamente
    print("False positive:", false_positive) # Stessa persona (con e senza mascherina) classificata scorrettamente
    print("True negative:", true_negative) # Diversa persona (con e senza mascherina) classificata correttamente
    print("False negative:", false_negative)  # Diversa persona (con e senza mascherina) classificata scorrettamente

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    f1_score = 2 * (precision*recall)/(precision+recall)

    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 Score:", f1_score)

    print("Masked Non Detected Face:", masked_non_detected_counter)
    print("UnMasked Non Detected Face:", unmasked_non_detected_counter)

    print("Finish")
    print("*******************************************************************")


if __name__ == '__main__':
    main()