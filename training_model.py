import os
import sys
import face_recognition
import pickle
import cv2

# функция для расчета енкодинга найденного лица
def training_model_by_img(name):

    if not os.path.exists("datasets"):
        print("[ERROR] there is not directory 'dataset'")
        sys.exit()

    know_encoding = []
    images = os.listdir('datasets')

    # print(images)
    for(i, image) in enumerate(images):
        print(f'[+] processing img {i + 1}/{len(images)}')
        print(image)
        face_img = face_recognition.load_image_file(f'datasets/{image}')
        face_loc = face_recognition.face_locations(face_img)
        if face_loc:
            pass
        else:
            continue
        face_enc = face_recognition.face_encodings(face_img)[0]

        # print(face_enc)

        if len(know_encoding) == 0:
            know_encoding.append(face_enc)
        else:
            for item in range(0, len(know_encoding)):
                result = face_recognition.compare_faces([face_enc], know_encoding[item])
                # print(result)
                if result[0]:
                    know_encoding.append(face_enc)
                    # print("Same person!")
                    break
                else:
                    # print("Another person!")
                    break
    # print(know_encoding)
    # print(f'Length {len(know_encoding)}')

    data = {
        "name": name,
        "encodings": know_encoding
    }
    print(data)
    with open(f"{name}_encidings.pickle", "wb") as file:
        file.write(pickle.dumps(data))

    return f"[INFO] File {name}_encodings.pickle successfuly created"
# функция для сбора фота из потока видео
def take_screenshot_from_video():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0

    if not os.path.exists("datasets_frame"):
        os.mkdir("datasets_frame")

    while True:
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        # multiplier = fps * 3
        # print(fps)

        if ret:
            # frame_id = int(round(cap.get(1)))
            # print(frame_id)
            cv2.imshow("frame", frame)
            k = cv2.waitKey(20)

            # if frame_id % multiplier == 0:
            #     cv2.imwrite(f'datasets_from_video/{count}.jpg', frame)
            #     count += 1
            if k == ord(" "):
                cv2.imwrite(f'datasets_frame/{count}.jpg', frame)
                print(f'Take an screenshot {count}')
                count += 1
            if k == ord("q"):
                break

        else:
            print("[ERROR] Can't get the frame...")
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print(training_model_by_img("Veronika"))
    # take_screenshot_from_video()

if __name__ == '__main__':
    main()