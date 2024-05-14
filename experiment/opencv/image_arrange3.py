import cv2
import numpy as np

# 글로벌 변수
refPt = []
cropping = False
image = None


def click_and_crop(event, x, y, flags, param):
    global refPt, cropping, image

    # 마우스 왼쪽 버튼 누를 시 시작 좌표 지정
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # 마우스 왼쪽 버튼 놓을 때까지 좌표 지정
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


def main():
    global image
    # 이미지 로드
    image_path = r"C:\Playground\experiment\opencv\test_image\interior_test.jpg"  # 이미지 경로 설정
    image = cv2.imread(image_path)
    clone = image.copy()
    mask = np.zeros(image.shape[:2], dtype="uint8")  # 마스크 초기화
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # 'r' 키를 누르면 이미지를 리셋
        if key == ord("r"):
            image = clone.copy()
            mask = np.zeros(image.shape[:2], dtype="uint8")

        # 'c' 키를 누르면 선택 영역을 잘라서 새 위치에 붙여넣기
        elif key == ord("c"):
            if len(refPt) == 2:
                # ROI 잘라내기
                roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                cv2.imshow("ROI", roi)
                # 선택 영역을 검은색으로 채우기
                cv2.rectangle(mask, refPt[0], refPt[1], (255), -1)
                image = cv2.bitwise_and(clone, clone, mask=cv2.bitwise_not(mask))
                cv2.imshow("image", image)

                # 사용자가 새 위치를 마우스로 선택 (여기서는 새 위치를 입력 받음)
                print("new X:")
                x_new = int(input())
                print("new Y:")
                y_new = int(input())

                # 새 위치에 ROI 붙여넣기
                if y_new + roi.shape[0] < image.shape[0] and x_new + roi.shape[1] < image.shape[1]:
                    image[y_new:y_new + roi.shape[0], x_new:x_new + roi.shape[1]] = roi
                cv2.imshow("image", image)

        # 's' 키를 누르면 이미지 저장
        elif key == ord("s"):
            cv2.imwrite(r"C:\Playground\experiment\opencv\arranged_image\arranged_image.jpg", image)

        # 'q' 키를 누르면 종료
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()