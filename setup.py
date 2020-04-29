from pathlib import Path
import os
import requests
import tarfile

# Make data directory and move inside
path_name = 'data/'
Path("data/").mkdir(parents=True, exist_ok=True)
os.chdir(path_name)
print('\nData directory created...')


# Downlad Data tar.gz
print('cifar-10-python.tar.gz download starting...')
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
target_path = 'cifar-10-python.tar.gz'
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(target_path, 'wb') as f:
        f.write(response.raw.read())
print('cifar-10-python.tar.gz download complete...')

# Unpack tar.gz
print('Unpacking files...')
tar = tarfile.open("cifar-10-python.tar.gz")
tar.extractall()
tar.close()

# Remove tar.gz file
print('Cleanup removing tarfile...')
os.remove("cifar-10-python.tar.gz")
print('\nAll done, your data is ready to go!!!\n')
