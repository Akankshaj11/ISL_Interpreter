import cv2

def draw_overlay_text(frame, recognized_text, running):
    status = "Running" if running else "Paused"
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Detected: {recognized_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, "Press S: Start | P: Pause | Q: Quit", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
