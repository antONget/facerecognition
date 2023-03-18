import face_recognition
from PIL import Image, ImageDraw
import pickle
import cv2
import tensorflow
from collections import defaultdict
import numpy as np
from mtcnn import MTCNN

def face_rec():
    zveri_face_img = face_recognition.load_image_file('images/zveri_3.jpg')
    zveri_face_img_location = face_recognition.face_locations(zveri_face_img)
    print(zveri_face_img_location)
    print(f"Found {len(zveri_face_img_location)} face(s) in this image")

    pil_img = Image.fromarray(zveri_face_img)
    draw = ImageDraw.ImageDraw(pil_img)

    for (top, right, bottom, left) in zveri_face_img_location:
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4)

    del draw
    pil_img.save('images/new_zveri_1.jpg')

def extacting_faces(img_path):
    count = 0
    faces = face_recognition.load_image_file(img_path)
    faces_locations = face_recognition.face_locations(faces)

    for faces_location in faces_locations:
        top, right, bottom, left = faces_location

        face_img = faces[top:bottom, left:right]
        pil_img = Image.fromarray(face_img)
        pil_img.save(f'images/{count}_face_img.jpg')
        count += 1

    return f"Found {count} face(s) in this photo"

def compare_faces(img1_path, img2_path):
    img1 = face_recognition.load_image_file(img1_path)
    img1_encoding = face_recognition.face_encodings(img1)[0]
    # print(img1_encoding)
    img2 = face_recognition.load_image_file(img2_path)
    img2_encoding = face_recognition.face_encodings(img2)[0]
    # print(img2_encoding)

    result = face_recognition.compare_faces([img1_encoding], img2_encoding)
    print(result)


def detect_person_in_video(add_person = 1):
# функция обнаруживает и распазнает лица в видеопатоке

    # загрузка encoding-вектора известных лиц
    # data = {}
    # list_filename_encoding = os.listdir('Encodings_faces_persons')
    # for filename_enciding in list_filename_encoding:
    #     filename_enciding_path = 'Encodings_faces_persons/'+filename_enciding
    #     data_face_person = pickle.loads(open(filename_enciding_path, "rb").read())
    #     data[data_face_person['name']] = data_face_person['encodings']


    data = pickle.loads(open('data_encidings.pickle', "rb").read())
    persons = data.keys()
    video = cv2.VideoCapture(0)

    while True:
        ret, image = video.read()
        # cv2.imshow('detect_person_in _video', image)
        locations = face_recognition.face_locations(image, model='hog')

        encodings = face_recognition.face_encodings(image, locations)
        person_unknown_loc = []
        person_unknown_enc = []
        for face_encoding, face_location in zip(encodings, locations):
            match = None
            for person in persons:
                result = face_recognition.compare_faces(data[person], face_encoding, tolerance=0.5)

                color = [0, 0, 255]
                if any(result):
                    match = person
                    print(f'Hi {match}')
                    color = [0, 255, 0]
                    break
            if match == None:
                print('Who is it?')
                person_unknown_loc.append(face_location)
                person_unknown_enc.append(face_encoding)

            left_top = (face_location[3], face_location[0])
            right_bottom = (face_location[1], face_location[2])
            # color = [0, 255, 0]
            cv2.rectangle(image, left_top, right_bottom, color, 4)

            left_bottom = (face_location[3], face_location[2])
            right_bottom = (face_location[1], face_location[2] + 20)
            cv2.rectangle(image, left_bottom, right_bottom, color, cv2.FILLED)
            cv2.putText(
                image,
                match,
                (face_location[3] + 10, face_location[2] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                4
            )

        cv2.imshow('detect_person_in _video', image)
        k = cv2.waitKey(1)
        print(k)
        if cv2.waitKey(233) & 0xFF == ord('q'):
            with open(f"data_encidings.pickle", "wb") as file:
                file.write(pickle.dumps(data))
            break
        if add_person:
            data = face_to_encoding(image, person_unknown_loc, person_unknown_enc, data)
            with open(f"data_encidings.pickle", "wb") as file:
                file.write(pickle.dumps(data))
def face_to_encoding(image, face_location, face_encoding, data):

    face_location = np.array(face_location)
    face_encoding = np.array(face_encoding)

    # data = {}#defaultdict(list)
    # list_filename_encoding = os.listdir('Encodings_faces_persons')
    # for filename_enciding in list_filename_encoding:
    #     filename_enciding_path = 'Encodings_faces_persons/'+filename_enciding
    #     data_face_person = pickle.loads(open(filename_enciding_path, "rb").read())
    #
    #     data[data_face_person['name']] = data_face_person['encodings']

    for loc in face_location:

        left_top = (loc[3], loc[0])
        right_bottom = (loc[1], loc[2])
        color = [255, 0, 0]
        cv2.rectangle(image, left_top, right_bottom, color, 4)
        cv2.imshow('detect_person_in _video', image)
        k = cv2.waitKey(1)
        name_person = input("Введите имя неизвестной персоны: ")

        # print(data.keys())

        flag = 1
        for key in data.keys():
            if name_person == key:
                temp_enc = data[name_person]
                temp_enc = np.vstack((temp_enc, face_encoding))
                # print(face_encoding)
                # print(type(face_encoding))
                # temp_enc.insert(-1, face_encoding)
                data[name_person] = temp_enc
                flag = 0
                break
        if flag:
            data[name_person] = face_encoding

    # print('---')
    # print(data.keys())


    # for key, value in data:
    #     if key == name_person:
    #         data[name_person].append(face_encoding)
    #         flag = 1
    #         break
    # if flag == 0:
    #     data[name_person].append(face_encoding)
    return data


def detect_person_in_video_mcnn(add_person=0):
    # функция обнаруживает и распазнает лица в видеопатоке

    # загрузка encoding-вектора известных лиц
    # data = {}
    # list_filename_encoding = os.listdir('Encodings_faces_persons')
    # for filename_enciding in list_filename_encoding:
    #     filename_enciding_path = 'Encodings_faces_persons/'+filename_enciding
    #     data_face_person = pickle.loads(open(filename_enciding_path, "rb").read())
    #     data[data_face_person['name']] = data_face_person['encodings']

    data = pickle.loads(open('data_encidings.pickle', "rb").read())
    persons = data.keys()
    video = cv2.VideoCapture(0)
    detector = MTCNN()
    while True:
        ret, image = video.read()
        # cv2.imshow('detect_person_in _video', image)
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        result = detector.detect_faces(image)
        locations = face_recognition.face_locations(image, model='hog')

        # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
        bounding_box = result[0]['box']
        keypoints = result[0]['keypoints']

        cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255),
                      2)

        cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

        cv2.imshow('detect_person_in _video', image)
def main():
    # face_rec()
    # print(extacting_faces("images/zveri_3.jpg"))
    # compare_faces('images/roma_1.jpg', 'images/roma_2.jpg')
    # detect_person_in_video_mcnn()
    detect_person_in_video(1)
if __name__ == '__main__':
    main()