import cv2
import numpy as np

# Global variables
refPt = []
cropping = False
image = None

def click_and_crop(event, x, y, flags, param):
    global refPt, cropping, image

    # Left mouse button pressed, start cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # Left mouse button released, end cropping
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        # Draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

def main():
    global image
    # Load image
    image_path = r"C:\Playground\experiment\opencv\test_image\interior_test.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found")
        return

    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # Reset the image when 'r' is pressed
        if key == ord("r"):
            image = clone.copy()

        # Crop the region of interest when 'c' is pressed
        elif key == ord("c"):
            if len(refPt) == 2:
                roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                cv2.imshow("ROI", roi)

                print("Drag the image to a new position using the mouse.")
                # Wait for a new mouse click for new position
                new_ref = cv2.selectROI("image", clone, fromCenter=False)
                x_offset, y_offset = int(new_ref[0]), int(new_ref[1])

                # Move the cropped ROI to the new position
                height, width = roi.shape[:2]
                if y_offset + height < clone.shape[0] and x_offset + width < clone.shape[1]:
                    clone[y_offset:y_offset + height, x_offset:x_offset + width] = roi

                cv2.imshow("Modified", clone)
                # Save the final image
                save_path = r"C:\Playground\experiment\opencv\arranged_image\arranged_image.jpg"
                cv2.imwrite(save_path, clone)

        # Quit the program when 'q' is pressed
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()