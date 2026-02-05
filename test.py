import time
import pyautogui

CLICK_INTERVAL = 60        # seconds (1 minute)
TOTAL_DURATION = 2 * 3600  # 2 hours in seconds

end_time = time.time() + TOTAL_DURATION

print("Clicker started. Move mouse to target position.")

while time.time() < end_time:
    pyautogui.click()
    print("Clicked")
    time.sleep(CLICK_INTERVAL)

print("Done. 2 hours completed.")
