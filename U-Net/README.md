## Setup Environemnets:

* create your virtual environment and install dependencies: 
  ```
  1. Open a terminal or command prompt.
  2. Navigate to the directory where you want to create the virtual environment (usually the path contains your script).
  3. Run the following command to create a new virtual environment: 
     /usr/bin/python3.6 -m venv venv
  4. Activate the virtual environment:
     source /.../venv/bin/activate
  6. Once the virtual environment is activated, run the following command to install the dependencies from the provided requirements.txt file:
     pip install --upgrade pip
     pip install -r requirements.txt
          or pip install numpy tensorflow imageio scipy tifffile
  ```
# Reference
    The tensorflow implementation, [https://github.com/lzhengchun/TomoGAN](https://github.com/lzhengchun/TomoGAN)
