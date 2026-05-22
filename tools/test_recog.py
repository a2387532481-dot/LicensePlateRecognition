import sys, os
sys.path.insert(0, 'D:\\University\\PythonProject2')
from plate_recognition.main import LicensePlateSystem

system = LicensePlateSystem(device='cpu')
system.initialize()
print('Model loaded.')

for car in ['car.jpg', 'car2.jpg', 'car3.jpg', 'car4.jpg']:
    path = os.path.join('D:\\University\\PythonProject2', car)
    if not os.path.isfile(path):
        print(f'{car}: not found')
        continue
    result = system.recognize(path, visualize=False)
    print(f'{car}: "{result["plate_text"]}"')
