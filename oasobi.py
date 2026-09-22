import subprocess

frame = subprocess.run(
        ["chafa", "--size=90x45", "image.png"],
        capture_output=True, text=True, check=True
).stdout

print(frame)