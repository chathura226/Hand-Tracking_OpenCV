# following can be used to change the volume in debian


import subprocess
import time

for i in range(0, 100):
    # Define the command
    command = f'amixer sset "Master" {str(i)}%'

    # Execute the command using subprocess
    subprocess.run(command, shell=True)
    time.sleep(1)
