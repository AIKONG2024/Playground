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
    image_path = r"C:\Playground\experiment\opencv\test_image\interior_test.jpg"
    image = cv2.imread(image_path)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # 'r' 키를 누르면 이미지를 리셋
        if key == ord("r"):
            image = clone.copy()

        # 'c' 키를 누르면 선택 영역을 잘라서 새 위치에 붙여넣기
        elif key == ord("c"):
            if len(refPt) == 2:
                # ROI extraction
                roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                mask = np.zeros(clone.shape[:2], dtype="uint8")
                cv2.rectangle(mask, refPt[0], refPt[1], 255, -1)
                blurred_background = cv2.GaussianBlur(clone, (21, 21), 0)
                clone = cv2.bitwise_and(blurred_background, blurred_background, mask=mask)
                mask_inv = cv2.bitwise_not(mask)
                foreground = cv2.bitwise_and(image, image, mask=mask_inv)
                clone = cv2.add(clone, foreground)

                # Clear the rectangle after processing
                image = clone.copy()  # Update the main image to remove the rectangle
                cv2.imshow("image", image)  # Refresh the image display

                # Get new location from user input
                x_new = int(input("Enter new X coordinate: "))
                y_new = int(input("Enter new Y coordinate: "))

                if y_new + roi.shape[0] < image.shape[0] and x_new + roi.shape[1] < image.shape[1]:
                    clone[y_new:y_new + roi.shape[0], x_new:x_new + roi.shape[1]] = roi
                cv2.imshow("Modified", clone)

        # 's' 키를 누르면 이미지 저장
        elif key == ord("s"):
            cv2.imwrite(r"C:\Playground\experiment\opencv\arranged_image\arranged_image.jpg", clone)

        # 'q' 키를 누르면 종료
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()