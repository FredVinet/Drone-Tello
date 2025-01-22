from djitellopy import Tello

tello = Tello()

try:
    tello.connect()
    print("Connected to Tello")
except Exception as e:
    print(f"Failed to connect to Tello: {e}")
    tello.end()
    exit(1)

print(tello.get_battery(), "%")

tello.takeoff()

pad = tello.get_mission_pad_id()
print(f"Mission pad number is: {pad}")

if pad <= 3:
    tello.rotate_clockwise(180)
else:
    tello.rotate_clockwise(360)

tello.disable_mission_pads()
tello.land()
tello.end()