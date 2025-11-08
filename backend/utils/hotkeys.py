def handle_keypress(key, running):
    if key == ord('s'):
        print("▶️ Started Detection")
        return True, None
    elif key == ord('p'):
        print("⏸️ Paused Detection")
        return False, None
    elif key == ord('q'):
        print("🛑 Exiting...")
        return running, "quit"
    return running, None
